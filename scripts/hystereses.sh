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
ZGN_steps=10000
ZGN_ns="64 128 256"
ZGN_cs="0.3 0.4 0.5"
ZGN_ds="0.05 0.1 0.15"
ZGN_as="0 0.1 0.2"
ZGN_ns="32"
ZGN_cs="0.5 1.0 2.0"
ZGN_ds="0.1"
ZGN_as="0 0.25 0.5"

jid=$((SLURM_ARRAY_TASK_ID-1))

for n in $ZGN_ns; do
for c in $ZGN_cs; do
for d in $ZGN_ds; do
for a in $ZGN_as; do

nr=`echo $n $c | awk '{printf("%d",$1*$2*log($1))}'`
nd=`bc <<< "${n}*${d} / 1"`
na=`bc <<< "${nr}*${a} / 1"`
echo $n $nr $nd $na

ZGN_filebase0="data/hystereses/${n}/${c}/${d}/${a}"
mkdir -p $ZGN_filebase0

for sid in `seq $ZGN_num`; do
seed=$((ZGN_num*jid+sid))
ZGN_filebase="${ZGN_filebase0}/${seed}"
./rmtchem.py --filebase $ZGN_filebase --n $n --nr $nr --nd $nd --na $na --seed $seed --steps $ZGN_steps --skip $ZGN_skip &

js=`jobs | wc -l`
while [ $js -ge 16 ]; do
  sleep 1
  js=`jobs | wc -l`
done
done

done
done
done
done

wait
