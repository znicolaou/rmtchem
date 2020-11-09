#!/bin/bash
#SBATCH -n 16
#SBATCH -a [1-5]
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
ZGN_num=1000
ZGN_gnum=1
ZGN_max=1

jid=$((SLURM_ARRAY_TASK_ID))
ZGN_n=$((100*jid))
filebase0=data/${ZGN_c}_${ZGN_n}_2
mkdir -p $filebase0
for sid in `seq $ZGN_num`; do
seed=$((sid))
echo seed is $seed n in $ZGN_n
filebase=${filebase0}/$seed

if [ ! -f ${filebase}evals.npy ]; then
./rmt.py --zr0 0 --zr1 0 --zi0 $ZGN_max --zi1 $ZGN_max --gnum $ZGN_gnum --n ${ZGN_n} --c $ZGN_c --mu $ZGN_mu --sigma $ZGN_sigma --seed $seed --filebase $filebase --output 1 &
else
echo "Previously run $seed"
fi

js=`jobs | wc -l`
while [ $js -ge 16 ]; do
  sleep 1
  js=`jobs | wc -l`
done

done
wait
