import os
import time

import numpy as np

_MPI_CHECKED = False
_COMM = None


def _get_communicator():
    global _MPI_CHECKED, _COMM
    if not _MPI_CHECKED:
        try:
            from mpi4py import MPI
        except ImportError:
            _COMM = None
        else:
            comm = MPI.COMM_WORLD
            _COMM = comm if comm.Get_size() > 1 else None
        _MPI_CHECKED = True
    return _COMM


def is_mpi_available():
    return _get_communicator() is not None


def _get_rank():
    comm = _get_communicator()
    return comm.Get_rank() if comm else 0


def get_size():
    comm = _get_communicator()
    return comm.Get_size() if comm else 1


def is_master():
    return _get_rank() == 0


def print_master(message, end="\n"):
    if is_master():
        print(message, end=end, flush=True)


def bcast(obj, root=0):
    comm = _get_communicator()
    return comm.bcast(obj, root=root) if comm else obj


# ---------------------------------------------------------------------------
# Idle waiting
#
# A blocking MPI call polls the network in a tight loop, so a rank waiting inside
# comm.bcast/scatter holds its core at 100%. That is invisible while every rank is
# evaluating, but the workers are idle for the WHOLE of training, sampling and
# acquisition -- the majority of an iteration once the likelihood is cheap -- and
# during that stretch seven spinning ranks compete with rank 0's multi-threaded
# Keras training for the same cores.
#
# Measured on the 8-core M3, rank 0 training a 5x512 network for 30 epochs while the
# other seven ranks waited for it:
#
#     1 rank, nobody waiting                      9.15 s   1.00x
#     8 ranks, blocking collective               20.03 s   2.19x   <- the bug
#     8 ranks, blocking + mpi_yield_when_idle=1  11.36 s   1.24x
#     8 ranks, idle_wait below                   10.30 s   1.13x
#     8 ranks, idle_wait + mpi_yield_when_idle=1  9.39 s   1.03x
#
# So the long waits are replaced by a non-blocking receive polled with sleep in
# between. Sleeping costs one poll interval of latency twice per iteration, against
# roughly half the training wall clock.
#
# The environment variable is complementary, not redundant -- it also helps the
# short per-batch waits inside the evaluation loop, which this cannot reach -- and
# the two together are essentially free. Open MPI only:
#
#     export OMPI_MCA_mpi_yield_when_idle=1
# ---------------------------------------------------------------------------

_WAKE_TAG = 7301
_IDLE_POLL_SECONDS = 0.05


def _idle_wait_for_master(comm, tag=_WAKE_TAG, root=0, poll=_IDLE_POLL_SECONDS,
                          poll_max=None):
    """Block until `root` releases us, sleeping rather than spinning.

    `poll` is the first sleep and `poll_max` the ceiling it doubles towards. The
    default is a flat 50 ms, which is right for the long between-phase wait where
    latency is irrelevant. The dynamic dispatcher's work loop passes a much smaller
    starting value: there the wait is a HOT path, and a flat 50 ms would add half of
    itself to the turnaround of every single point.
    """
    poll_max = poll if poll_max is None else poll_max
    request = comm.irecv(source=root, tag=tag)
    interval = poll
    while True:
        received, payload = request.test()
        if received:
            return payload
        time.sleep(interval)
        interval = min(interval * 2.0, poll_max)


def _release_idle_workers(comm, payload, tag=_WAKE_TAG):
    """Wake every worker parked in _idle_wait_for_master."""
    for destination in range(1, comm.Get_size()):
        comm.send(payload, dest=destination, tag=tag)


def bcast_idle(obj, root=0):
    """bcast for a value the workers may have to wait a long time for.

    Same contract as bcast, but the workers sleep instead of spinning. Use it
    wherever the master does substantial work before the broadcast; use plain
    bcast when the wait is short, since this one costs a poll interval.
    """
    comm = _get_communicator()
    if comm is None:
        return obj
    if comm.Get_rank() == root:
        _release_idle_workers(comm, obj)
        return obj
    return _idle_wait_for_master(comm, root=root)


def _format_progress(done, total, elapsed):
    percent = 100.0 * done / total if total else 100.0
    return f"Evaluated {done}/{total} samples ({percent:.1f}%) in {elapsed:.2f}s"


def _progress_batch_size(n_processes):
    return max(1, n_processes)


def _evaluate_local(samples, evaluator):
    return np.asarray([evaluator(sample) for sample in samples])


# ---------------------------------------------------------------------------
# Work distribution
#
# The original scheme split each batch of n_ranks points one-per-rank, then
# gathered. A gather is a barrier, so EVERY point cost the SLOWEST rank's time and
# the faster ranks idled for the difference. That is invisible on a uniform machine
# and brutal on a heterogeneous one.
#
# Measured on the 8-core M3 (4 performance + 4 efficiency cores) running the real
# Planck likelihood, which takes 1.87 s on a performance core:
#
#     2000-point initial design, 8 ranks, static:  1337 s = 1.50 evaluations/s
#     per-batch time 5.35 s -- i.e. the efficiency-core time, not the mean
#
# The four performance-core ranks were idle roughly 3.5 s in every 5.35 s. A short
# burst hides this completely: the same code on a 200-point design measured
# 3.63 evaluations/s, because macOS keeps a brief load on performance cores and
# only spreads a sustained one onto the efficiency cores.
#
# The dynamic scheme instead has rank 0 hand out one index at a time to whichever
# rank reports back next, so a slow rank simply completes fewer points and nobody
# waits on anybody. Throughput becomes the SUM of the per-rank rates rather than
# n_ranks times the slowest.
#
# IT IS OFF BY DEFAULT, BECAUSE ON THIS MACHINE THE PREMISE IS FALSE. The diagnosis
# above -- that the efficiency cores were holding every batch back -- was wrong, and
# the measurement that refutes it is the per-rank one: over a 640-point interleaved
# A/B on the real Planck likelihood, ranks 1-7 took
#
#     3.28  3.27  3.30  3.27  3.31  3.28  3.28   seconds per evaluation
#
# i.e. homogeneous to +/-0.6%. There is no fast rank and no slow rank, so there is
# nothing for load balancing to recover, and dedicating rank 0 to dispatch simply
# costs an eighth of the machine.
#
# The A/B itself could not separate the two arms either:
#
#     static   120.7 s then  47.6 s   (1.33 and 3.36 evaluations/s)
#     dynamic   66.0 s then  54.4 s   (2.43 and 2.94 evaluations/s)
#
# The within-arm spread is 2.5x and the arms overlap completely -- dynamic's whole
# range sits inside static's. That is not evidence of a difference, it is evidence
# that this machine's sustained throughput wanders by more than the effect being
# measured. What actually moved the production run from 3.63 to 1.50 evaluations/s
# was that same wander (a solo evaluation is 1.87 s, one of eight concurrent ones is
# 3.3 s or worse), not any imbalance between ranks.
#
# The residual argument for dynamic is real but small: a batch costs the MAX of its
# points and CLASS varies about +/-10% point to point, so static gives up roughly
# 13% -- almost exactly the 12.5% given up by reserving a dispatcher. A wash, as
# measured. It would only become a win if rank 0 evaluated as well as dispatched,
# which needs prefetching and is not worth the deadlock risk on a ten-hour run.
#
# So: kept, correct, documented, and opt-in.
#
# Rank 0 dispatches and does not evaluate. That costs one rank and buys the
# balance, which is a large net win whenever the ranks are not identical -- and it
# also removes the fragile alternative, where a master that evaluates cannot answer
# a request until its own point is finished. Below four ranks the trade stops paying
# (too few evaluators left, and few enough ranks that they all land on performance
# cores), so the static path is kept and used there.
#
# WHICH SCHEDULE IS RIGHT DEPENDS ON HOW EXPENSIVE THE LIKELIHOOD IS, and CLiENT's
# targets span six orders of magnitude: the analytic banana and Gaussian targets
# evaluate in microseconds, real Planck takes 1.9 s. Dispatching one point at a time
# costs an MPI round trip plus the dispatcher's poll latency, so on a microsecond
# likelihood dynamic scheduling is far SLOWER than static -- measured on a synthetic
# 64-point workload with 20-160 ms tasks, dynamic ran 1.42 s against static's 1.36 s
# purely on dispatch overhead, and the gap widens as the task shrinks.
#
# So the master times one point before choosing, and only goes dynamic when an
# evaluation is long enough to bury the dispatch cost. That probe is one evaluation
# per phase: 0.15% on Planck, and free on the targets where it might matter.
#
# Set CLIENT_MPI_SCHEDULE=static or =dynamic to force one, for A/B measurement.
#
# Results are identical either way: the point set is unchanged, loglkl is
# deterministic in x, and values are written back by index. Only WHICH rank
# evaluates a given point changes.
# ---------------------------------------------------------------------------

_WORK_TAG = 7302
_RESULT_TAG = 7303
# The dispatcher backs off from a near-immediate re-poll to this ceiling, so a result
# that is already queued costs nothing and an idle stretch costs almost no CPU.
_DISPATCH_POLL_MIN_SECONDS = 0.0005
_DISPATCH_POLL_MAX_SECONDS = 0.01
_MIN_RANKS_FOR_DYNAMIC = 4
# An evaluation must be worth at least this much for per-point dispatch to pay. At
# 20 ms the round trip and poll latency are roughly 5%; below it they take over.
_DYNAMIC_MIN_SECONDS = 0.02


def _requested_schedule():
    requested = os.environ.get("CLIENT_MPI_SCHEDULE", "").strip().lower()
    return requested if requested in ("static", "dynamic") else None


def _use_dynamic_schedule(size, seconds_per_point):
    """Whether dynamic dispatch is worth it. OFF unless forced -- see the note above.

    Kept as a function rather than inlined because the two conditions below are the
    ones that would have to hold for it to pay, and a future machine may satisfy
    them. Nothing calls this on the default path.
    """
    if size < _MIN_RANKS_FOR_DYNAMIC:
        return False
    return seconds_per_point >= _DYNAMIC_MIN_SECONDS


def _evaluate_static(comm, points, total, evaluator, start_time):
    """Lock-step: scatter one slice per rank, gather, repeat. Barrier per batch."""
    size = comm.Get_size()
    gathered_values = []
    batch_size = _progress_batch_size(size)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunks = np.array_split(points[start:end], size, axis=0) if is_master() else None
        local_samples = comm.scatter(chunks, root=0)
        local_values = _evaluate_local(local_samples, evaluator)
        batch_values = comm.gather(local_values, root=0)
        if is_master():
            gathered_values.append(np.concatenate(batch_values))
            print_master(_format_progress(end, total, time.monotonic() - start_time))

    if is_master() and gathered_values:
        return np.concatenate(gathered_values)
    return np.array([])


def _evaluate_dynamic(comm, points, total, evaluator, start_time, probe_value=None):
    """Rank 0 hands out indices on demand; no rank ever waits for another."""
    from mpi4py import MPI

    size = comm.Get_size()

    if not is_master():
        # points was broadcast, so only an index travels per assignment.
        while True:
            index = _idle_wait_for_master(
                comm,
                tag=_WORK_TAG,
                poll=_DISPATCH_POLL_MIN_SECONDS,
                poll_max=_DISPATCH_POLL_MAX_SECONDS,
            )
            if index is None:
                return np.array([])
            value = float(evaluator(points[index]))
            # The rank is in the payload so the dispatcher needs no MPI.Status.
            comm.send((comm.Get_rank(), index, value), dest=0, tag=_RESULT_TAG)

    values = np.empty(total, dtype=float)
    # The schedule probe already evaluated point 0; keep its value rather than
    # paying for it twice.
    next_index = 0
    completed = 0
    if probe_value is not None:
        values[0] = probe_value
        next_index = 1
        completed = 1
    progress_every = _progress_batch_size(size)

    for worker in range(1, size):
        if next_index < total:
            comm.send(next_index, dest=worker, tag=_WORK_TAG)
            next_index += 1

    while completed < total:
        # Sleep-poll rather than block: a spinning dispatcher would take a core back
        # off the evaluators, which is the whole point of bcast_idle above. Back off
        # from near-zero so a queued result is picked up immediately -- a fixed
        # interval adds half of itself to every point's turnaround.
        request = comm.irecv(source=MPI.ANY_SOURCE, tag=_RESULT_TAG)
        poll = _DISPATCH_POLL_MIN_SECONDS
        while True:
            received, payload = request.test()
            if received:
                break
            time.sleep(poll)
            poll = min(poll * 2.0, _DISPATCH_POLL_MAX_SECONDS)

        worker, index, value = payload
        values[index] = value
        completed += 1

        if next_index < total:
            comm.send(next_index, dest=worker, tag=_WORK_TAG)
            next_index += 1

        if completed % progress_every == 0 or completed == total:
            print_master(_format_progress(completed, total, time.monotonic() - start_time))

    for worker in range(1, size):
        comm.send(None, dest=worker, tag=_WORK_TAG)

    return values


def broadcast_and_evaluate(samples, evaluator):
    """Evaluate master-owned samples across MPI ranks."""
    comm = _get_communicator()
    if comm is None:
        points = np.asarray(samples)
        total = len(points)
        print_master(f"Evaluating {total} samples...")
        start_time = time.monotonic()
        values = _evaluate_local(points, evaluator)
        print_master(_format_progress(total, total, time.monotonic() - start_time))
        return points, values

    if is_master():
        points = np.asarray(samples)
        total = len(points)
        print_master(f"Evaluating {total} samples with {get_size()} MPI processes...")
    else:
        points = np.array([])
        total = None

    # The workers have been idle since the previous evaluation -- through training,
    # sampling and acquisition -- so this is the long wait, not a collective between
    # two busy ranks. Hand the count over without spinning; see bcast_idle above.
    total = bcast_idle(total, root=0)
    if total == 0:
        return points, np.array([])

    start_time = time.monotonic()

    # Static unless explicitly asked for otherwise. No probe on the default path, so
    # the schedule choice costs nothing.
    dynamic = _requested_schedule() == "dynamic"
    probe_value = None

    if dynamic:
        # Every rank needs the points, since assignments travel as bare indices.
        # At 2000 x 27 float64 this is 432 kB, sent once per evaluation phase.
        points = comm.bcast(points, root=0)
        values = _evaluate_dynamic(
            comm, points, total, evaluator, start_time, probe_value=probe_value
        )
    else:
        values = _evaluate_static(comm, points, total, evaluator, start_time)

    if is_master():
        return points, values
    # Under the dynamic schedule a worker holds the broadcast copy of every point.
    # Callers ignore a worker's return value, so hand back the same empty pair the
    # static path always did rather than leaking the full array.
    return np.array([]), np.array([])
