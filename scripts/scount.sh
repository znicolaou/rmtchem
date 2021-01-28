if [ $# -ne 1 ]; then
echo usage ./scount.sh filebase
exit
fi
filebase=$1
ns=`ls -d $filebase/*/ | cut -d/ -f3`

for n in $ns; do
  echo $n
  if [ -f ${filebase}/${n}_counts.txt ]; then rm ${filebase}/${n}_counts.txt; fi
  cs=`ls -d ${filebase}/${n}/*/ | cut -d/ -f4`
  for c in $cs; do
    echo $n $c
    echo $n $c `tail -n1 -q ${filebase}/${n}/${c}/* | awk -v n0=$n '{n+=$3/n0; t++; r+=$4/n0; if($3==n0){n2++}; if($4==n0){r2++}; }END{print(n/t,r/t,n2/t,r2/t)}'` >> ${filebase}/${n}_counts.txt
  done
done
