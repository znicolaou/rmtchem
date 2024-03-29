#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -a [1-16]
#SBATCH --time=07-00:00:00
#SBATCH --mem=120gb
#SBATCH --output=outs/%j.out
#SBATCH -p GTX980,K80

#module load python/anaconda3.7
#source activate my_env
export OMP_NUM_THREADS=1

if [ $# -ne 2 ]; then
echo usage ./hystereses.sh filebase atoms
exit
fi

ZGN_proc=16
ZGN_num=4096
ZGN_skip=10
ZGN_ns="128"
ZGN_cs="0.5 1.0 1.5"
#ZGN_ds="0.1 0.2 0.3"
ZGN_ds="0.05 0.15 0.25"
ZGN_as="0 0.25 0.5"
ZGN_natoms=$2
ZGN_filebase0=$1

for n in $ZGN_ns; do
for c in $ZGN_cs; do
for d in $ZGN_ds; do
for a in $ZGN_as; do

nr=`echo $n $c | awk '{printf("%d",$1*$2*log($1))}'`
nd=`bc <<< "${n}*${d} / 1"`
na=`bc <<< "${nr}*${a} / 1"`
echo $n $nr $nd $na

ZGN_filebase1="${ZGN_filebase0}/${n}/${c}/${d}/${a}"
mkdir -p $ZGN_filebase1

for seed in `seq $ZGN_num`; do
ZGN_filebase="${ZGN_filebase1}/${seed}"

if [ ! -f ${ZGN_filebase}out.dat ]; then
  #timeout 7200 ./rmtchem.py --filebase $ZGN_filebase --n $n --nr $nr --nd $nd --na $na --seed $seed --skip $ZGN_skip --atoms $ZGN_natoms &> /dev/null &
  timeout 7200 ./rmtchem.py --filebase $ZGN_filebase --n $n --nr $nr --nd $nd --na $na --seed $seed --skip $ZGN_skip --atoms $ZGN_natoms &> /dev/null &
fi

js=`jobs | wc -l`
while [ $js -ge $ZGN_proc ]; do
  sleep 1
  js=`jobs | wc -l`
done
done

done
done
done
done

wait
