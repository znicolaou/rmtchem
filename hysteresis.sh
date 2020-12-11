#!/bin/bash
#SBATCH -n 8
#SBATCH -a [1-30]
#SBATCH --time=00-2:00:00
#SBATCH --mem=16gb
#SBATCH --output=outs/%j.out
#SBATCH -p GTX980

module load python/anaconda3.7
source activate my_env
export OMP_NUM_THREADS=1

ZGN_num=8
ZGN_n=200
ZGN_nr=$((2*ZGN_n))
ZGN_nd=$((ZGN_n/10))
ZGN_filebase0="data/$ZGN_n"
mkdir -p $ZGN_filebase0

jid=$((SLURM_ARRAY_TASK_ID-1))

for sid in `seq $ZGN_num`; do
seed=$((ZGN_num*jid+sid))
ZGN_filebase="${ZGN_filebase0}/${seed}"
#echo filebase is $ZGN_filebase
./rmtchem.py --filebase $ZGN_filebase --n $ZGN_n --nr $ZGN_nr --nd $ZGN_nd --seed $seed --steps 1000 &

js=`jobs | wc -l`
while [ $js -ge 8 ]; do
  sleep 1
  js=`jobs | wc -l`
done

done
wait
