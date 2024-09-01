#!/bin/bash
#!/usr/bin/python3.9
#SBATCH --mail-user=m.hgog@campus.lmu.de
#SBATCH --mail-type=FAIL
#SBATCH --job-name=BABA
#SBATCH --output=BABA.out
#SBATCH --error=BABA.err
#SBATCH --partition=All
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1000:00:00
#SBATCH --mem=1GB

python3 data_baba.py