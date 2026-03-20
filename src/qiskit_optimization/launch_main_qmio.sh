#!/bin/bash
#SBATCH --job-name=main_qmio
#SBATCH --partition=qpu
#SBATCH --output=output.log
#SBATCH --error=error.log

conda deactivate
module load qmio/hpc  gcc/12.3.0  matplotlib jupyter-bundle qmio-run  matplotlib
module load qiskit  qmio-tools

python -u main_qmio.py