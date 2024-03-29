export OMP_NUM_THREADS=1

procs=16
ZGN_num=1024
ZGN_ns="64 256 1024"
#ZGN_ns="64 128 256"
ZGN_cmin=0.08125
ZGN_cmax=1.625
ZGN_dc=0.08125


for seed in `seq 1 $ZGN_num`; do
echo $seed
for c in `seq $ZGN_cmin $ZGN_dc $ZGN_cmax`; do
for n in $ZGN_ns; do
nr=`bc <<< "${n}*${c} / 1"`
ZGN_filebase0="data/er/${n}/${c}"
mkdir -p $ZGN_filebase0
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
