"""Deterministic tests for learning-rate annealing and its schedule-gated stopper.

These do NOT belong in the smoke run. On an easy target val_loss improves every epoch, so
ReduceLROnPlateau never detects a plateau and never reduces -- a smoke test would pass
while exercising nothing. These tests drive the callbacks directly on a contrived loss
sequence, so a reduction and a stop are guaranteed to occur.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.training import StopAfterScheduleExhausted, build_callbacks


def _model():
    m = tf.keras.Sequential([tf.keras.layers.Input((2,)), tf.keras.layers.Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="mse")
    return m


def test_build_callbacks_plateau_and_none():
    cbs = build_callbacks("plateau", 0.3, 5, 1e-6, 10, 250)
    kinds = [type(c).__name__ for c in cbs]
    assert kinds == ["ReduceLROnPlateau", "StopAfterScheduleExhausted"], kinds
    cbs = build_callbacks("none", 0.3, 5, 1e-6, 10, 250)
    assert [type(c).__name__ for c in cbs] == ["EarlyStopping"]
    try:
        build_callbacks("cosine", 0.3, 5, 1e-6, 10, 250)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown lr_schedule should raise")
    print("  build_callbacks: plateau -> [ReduceLROnPlateau, gated stopper]; "
          "none -> [EarlyStopping]; unknown -> ValueError")


def test_stopper_waits_for_min_lr_then_stops():
    """The stopper must not fire while the rate is still above lr_min."""
    m = _model()
    s = StopAfterScheduleExhausted(lr_min=1e-6, grace=10, verbose=0)
    s.set_model(m)
    # rate still high: no stop, however long the loss stalls
    m.optimizer.learning_rate.assign(1e-3)
    for e in range(200):
        s.on_epoch_end(e, {"val_loss": 1.0})
    assert not m.stop_training, "stopped while lr was still above lr_min"
    assert s.epoch_at_min is None
    # rate reaches the floor: the grace period starts HERE, not at epoch 0
    m.optimizer.learning_rate.assign(1e-6)
    for e in range(200, 205):
        s.on_epoch_end(e, {"val_loss": 1.0})
    assert not m.stop_training, "stopped before the grace period elapsed"
    for e in range(205, 215):
        s.on_epoch_end(e, {"val_loss": 1.0})
    assert m.stop_training, "did not stop after the grace period"
    assert s.epoch_at_min == 200, s.epoch_at_min
    print("  stopper: ignored 200 stalled epochs above lr_min, then stopped "
          f"{215 - s.epoch_at_min - 1} epochs after reaching it (grace 10)")


def test_stopper_survives_monotone_improvement():
    """The case a plain EarlyStopping(min_delta=0) cannot terminate."""
    m = _model()
    s = StopAfterScheduleExhausted(lr_min=1e-6, grace=20, verbose=0)
    s.set_model(m)
    m.optimizer.learning_rate.assign(1e-6)
    vl = 1.0
    for e in range(100):
        vl *= 0.999                     # improves EVERY epoch, forever
        s.on_epoch_end(e, {"val_loss": vl})
        if m.stop_training:
            break
    assert m.stop_training, "a monotonically improving curve must still terminate"
    assert e == 20, f"stopped at epoch {e}, expected 20"
    print(f"  stopper: terminated a monotonically improving curve at epoch {e} "
          "(a min_delta=0 EarlyStopping never would)")


def test_stopper_restores_best_weights():
    m = _model()
    s = StopAfterScheduleExhausted(lr_min=1e-6, grace=2, verbose=0)
    s.set_model(m)
    m.optimizer.learning_rate.assign(1e-6)
    s.on_epoch_end(0, {"val_loss": 0.5})            # best
    good = [w.copy() for w in m.get_weights()]
    m.set_weights([w + 1.0 for w in m.get_weights()])   # wander off
    s.on_epoch_end(1, {"val_loss": 9.0})
    s.on_epoch_end(2, {"val_loss": 9.0})
    s.on_train_end()
    assert all(np.allclose(a, b) for a, b in zip(m.get_weights(), good)), \
        "best weights were not restored"
    assert s.best_epoch == 0
    print("  stopper: restored the best weights (epoch 1), not the final ones")


def test_reduction_and_stop_inside_a_real_fit():
    """Both callbacks inside a real fit loop, with the loss sequence CONTROLLED.

    An earlier version of this test hoped a fit on pure noise would plateau. It does not
    reliably: with min_delta=0 any fluctuation to a new minimum resets ReduceLROnPlateau's
    counter, so no reduction is guaranteed and the test was flaky. Here a callback ordered
    BEFORE the schedule overwrites val_loss with a constant, which makes the plateau
    certain while still exercising the real Keras callback machinery.
    """
    class ConstantValLoss(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs["val_loss"] = 1.0          # never improves, so the plateau is certain

    rng = np.random.default_rng(0)
    X = rng.standard_normal((128, 2)).astype(np.float32)
    y = rng.standard_normal(128).astype(np.float32)
    m = _model()
    # lr 1e-2 -> 1e-4 by factor 0.3 needs 4 reductions, 5 stalled epochs each, then grace 10
    cbs = [ConstantValLoss()] + build_callbacks("plateau", 0.3, 5, 1e-4, 10, 250)
    h = m.fit(X, y, epochs=200, batch_size=64, validation_split=0.25, verbose=0,
              callbacks=cbs)
    lr = np.array(h.history["learning_rate"])
    rates = sorted(set(np.round(lr, 10)), reverse=True)
    assert len(rates) > 1, f"learning rate never reduced: {rates}"
    assert rates[-1] <= 1e-4 * 1.01, f"did not reach lr_min: {rates}"
    assert len(lr) < 200, f"ran the full ceiling ({len(lr)}); the stopper never fired"
    print(f"  real fit: {len(rates)} distinct rates {[f'{r:.1e}' for r in rates]}, "
          f"reached lr_min, stopped at epoch {len(lr)} of a 200 ceiling")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}:")
            fn()
    print("\nall tests passed")
