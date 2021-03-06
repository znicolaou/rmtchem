#!/bin/bash
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -a [1-16]
#SBATCH --time=04-00:00:00
#SBATCH --mem=120gb
#SBATCH --output=outs/%j.out
#SBATCH -p GTX980

module load python/anaconda3.7
source activate my_env
export OMP_NUM_THREADS=1

procs=$SLURM_NTASKS
ZGN_num=64
ZGN_ns="64 512 4096"
ZGN_cmin=0.1
ZGN_cmax=6.6
ZGN_dc=0.1

jid=$((SLURM_ARRAY_TASK_ID-1))

for n in $ZGN_ns; do
for sid in `seq $ZGN_num`; do
echo $sid
for c in `seq $ZGN_cmin $ZGN_dc $ZGN_cmax`; do
nr=`bc <<< "${n}*${c} / 1"`
ZGN_filebase0="data/er/${n}/${c}"
mkdir -p $ZGN_filebase0
seed=$((ZGN_num*jid+sid))
ZGN_filebase="${ZGN_filebase0}/${seed}"
if [ -f ${ZGN_filebase}out.dat ]; then
echo already completed $ZGN_filebase
else
./rmtchem.py --type 1 --filebase $ZGN_filebase --n $n --nr $nr --seed $seed --quasistatic 0 --rank 1 &
fi
js=`jobs | wc -l`
while [ $js -ge $procs ]; do
  sleep 1
  js=`jobs | wc -l`
done
done

done
done

wait
