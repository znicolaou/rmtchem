if [ $# -ne 2 ]; then
echo usage ./scount.sh filebase atoms
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
    echo $n $c `tail -n1 -q ${filebase}/${n}/${c}/* | awk -v n0=$n -v na=$2 '{n+=$3/n0; t++; r+=$4/($3); l+=$14; if($3==n0){n2++}; if($4==($3-na)){r2++}; }END{print(n/t,r/t,n2/t,r2/t,l/t)}'` >> ${filebase}/${n}_counts.txt
  done
done
