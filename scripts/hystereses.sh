#!/bin/bash
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -a [1-16]
#SBATCH --time=01-00:00:00
#SBATCH --mem=16gb
#SBATCH --output=outs/%j.out
#SBATCH -p GTX980

module load python/anaconda3.7
source activate my_env
export OMP_NUM_THREADS=1

ZGN_num=64
ZGN_skip=10
ZGN_steps=5000
ZGN_ns="50 100 200 500 1000"
ZGN_cs="2 2.5 3"
ZGN_ds="25 10 5"

jid=$((SLURM_ARRAY_TASK_ID-1))

for n in $ZGN_ns; do
for c in $ZGN_cs; do
for d in $ZGN_ds; do
echo $n $c $d
#nr=$((n*c))
nr=`bc <<< "${n}*${c} / 1"`
nd=$((n/d))
ZGN_filebase0="data/hystereses/${n}/${c}/${d}"
mkdir -p $ZGN_filebase0

for sid in `seq $ZGN_num`; do
seed=$((ZGN_num*jid+sid))
ZGN_filebase="${ZGN_filebase0}/${seed}"
./rmtchem.py --filebase $ZGN_filebase --n $n --nr $nr --nd $nd --seed $seed --steps $ZGN_steps --skip $ZGN_skip &

js=`jobs | wc -l`
while [ $js -ge 16 ]; do
  sleep 1
  js=`jobs | wc -l`
done
done

done
done
done

wait
