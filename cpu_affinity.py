import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Retrieve the set of logical CPUs this process is allowed to run on
allowed_cpus = os.sched_getaffinity(0)

print(f"[Rank {rank}] Allowed CPU Cores: {allowed_cpus} (Count: {len(allowed_cpus)})")
