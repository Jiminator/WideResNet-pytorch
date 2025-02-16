#!/bin/bash
#SBATCH --mem=64G
#SBATCH --output="WideResNet.out"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS, 64 requests whole node
#SBATCH --partition=gpuA100x4    # <- one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcrn-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=WideResNet
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -t 02:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

echo "job is starting on `hostname`"
# Record start time
start_time=$(date +%s)

# run the container binary with arguments: python3 <program.py>
# --bind /projects/bbXX  # add to apptainer arguments to mount directory inside container
# apptainer run --nv --bind /scratch/bcrn/jshong /scratch/bcrn/jshong/python3.sif /bin/bash -c "python train.py --name run_4 --dataset cifar10 --layers 40 --widen-factor 4 --tensorboard"
apptainer run --nv --bind /scratch/bcrn/jshong /scratch/bcrn/jshong/python3.sif /bin/bash -c "python train.py --name run_5 --dataset cifar10 --layers 52 --widen-factor 38 --epochs 10 --tensorboard"
end_time=$(date +%s)
# Compute duration
duration=$((end_time - start_time))
echo "Job completed. Duration: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s"