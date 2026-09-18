# Environment for the real-Planck runs. SOURCE THIS FROM BASH, NOT ZSH:
#
#     bash
#     source env_planck.sh
#     mpirun -n 8 python client.py input/planck2018_lcdm.yaml -n planck_lcdm -i 16
#
# clik_profile.sh defines a shell function using ${!var} indirect expansion, which zsh
# rejects with "bad substitution". Sourcing it from zsh leaves clik half-configured and
# the failure surfaces much later, as an import error inside MontePython.

CLIENT_ROOT="/Users/au78087/kantest/client-dev-sth1509"
PUBLIC_ROOT="/Users/au78087/kantest/client_public"
CLIK_ROOT="${PUBLIC_ROOT}/resources/planck/code/plc_3.0/plc-3.01"

# CLASS++ 26.0.0 (AarhusCosmology/CLASSpp_public), installed out of tree so that
# clienv's own classy 3.3.4 stays intact and switching back is one edit.
#   1.9 s per Planck evaluation at one thread, against 7.3 s for CLASS 3.3.4.
# Comment this line out to run against stock CLASS instead.
CLASSPP_INSTALL="/Users/au78087/kantest/CLASSpp_public/_install"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate clienv

source "${CLIK_ROOT}/bin/clik_profile.sh"

if [ -n "${CLASSPP_INSTALL}" ]; then
  export PYTHONPATH="${CLASSPP_INSTALL}:${PYTHONPATH}"
fi

# One thread per rank. CLASS++ has a substantial serial section, so its thread scaling is
# poor (1.87 s at 1 thread, 0.71 s at 4, 0.58 s at 8) and running many single-threaded
# model computations concurrently beats running few wide ones: 8 ranks x 1 thread gives
# 4.2 evaluations/s, 1 rank x 8 threads gives 1.7.
export OMP_NUM_THREADS=1

# Stop idle ranks spinning against rank 0's training. utils/mpi_utils.py::bcast_idle
# handles the two long waits per iteration; this covers the short per-batch ones it
# cannot reach. Together they land within 3% of uncontended training time. See CHANGES.md
# section 8. Open MPI only.
export OMPI_MCA_mpi_yield_when_idle=1

export TF_CPP_MIN_LOG_LEVEL=3

cd "${CLIENT_ROOT}" || return 1

echo "clienv + clik + $( [ -n "${CLASSPP_INSTALL}" ] && echo 'CLASS++ 26.0.0' || echo 'CLASS 3.3.4' )"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}  OMPI_MCA_mpi_yield_when_idle=${OMPI_MCA_mpi_yield_when_idle}"
python -c "import classy; print('classy', getattr(classy, '__version__', '?'), '->', classy.__file__)"
