#!/bin/bash
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=32
#SBATCH --hint=nomultithread
#SBATCH --nodes=2
#SBATCH --time=00:30:00


module load intelpython3
module load h5py 


srun -u -n ${SLURM_NTASKS} -N ${SLURM_NNODES} --hint=nomultithread python HLS_BuildMask_mpi_hdf5.py --xrange 100 199 --yrange 10 12 --writers 4
