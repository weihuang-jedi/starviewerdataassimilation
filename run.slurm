#! /bin/sh
#SBATCH --job-name=bmatrix
#SBATCH --account=epic
#SBATCH --qos=batch
#SBATCH --partition=u1-compute
#SBATCH -t 06:00:00
#SBATCH --ntasks=1
#SBATCH -o log.out
##SBATCH --exclusive

EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
source ${EAGLEhome}/conda/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate anemoi

cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewerdataassimilation

python generate_bmatrix_from_analyses.py

