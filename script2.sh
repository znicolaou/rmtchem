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

ZGN_c=2
ZGN_mu=1
ZGN_sigma=1.0
ZGN_n=100
ZGN_num=345
ZGN_gnum=101
ZGN_max=5
filebase0=${ZGN_c}_${ZGN_n}_1

mkdir -p data/$filebase0

jid=$((SLURM_ARRAY_TASK_ID-1))
for sid in `seq $ZGN_num`; do
seed=$((ZGN_num*jid+sid))
echo seed is $seed
filebase=data/${filebase0}/$seed

if [ ! -f ${filebase}evals.npy ]; then
./rmt.py --zr0 -$ZGN_max --zr1 $ZGN_max --zi0 -$ZGN_max --zi1 $ZGN_max --gnum $ZGN_gnum --n ${ZGN_n} --c $ZGN_c --mu $ZGN_mu --sigma $ZGN_sigma --seed $seed --filebase $filebase --output 1 &
else
echo "Previously run $seed"
fi

js=`jobs | wc -l`
while [ $js -ge 8 ]; do
  sleep 1
  js=`jobs | wc -l`
done

done
wait
