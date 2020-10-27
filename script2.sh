#!/bin/bash
#SBATCH -n 8
#SBATCH -a [1-128]
#SBATCH --time=00-2:00:00
#SBATCH --mem=16gb
#SBATCH --output=outs/%j.out
#SBATCH -p GTX980

module load python/anaconda3.7
source activate my_env
export OMP_NUM_THREADS=1

ZGN_c=2
ZGN_n=50
mkdir -p data/${ZGN_c}_${ZGN_n}

jid=$((SLURM_ARRAY_TASK_ID))
for sid in `seq 0 7`; do
seed=$((8*jid+sid))
echo seed is $seed
filebase=data/${ZGN_c}_${ZGN_n}/$seed

if [ ! -f ${filebase}evals.npy ]; then
./rmt.py --n ${ZGN_n} --c $ZGN_c --seed $seed --filebase $filebase --output 1 &
else
echo "Previously run $seed"
fi

done
wait
