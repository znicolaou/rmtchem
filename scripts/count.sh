if [ $# -ne 1 ]; then
echo "usage ./count.sh filebase"
exit
fi
filebase=$1
ns="50"

for n in $ns; do
  cs=`ls -d ${filebase}/${n}/*/ | cut -d/ -f4`
  for c in $cs; do
    ds=`ls -d ${filebase}/${n}/${c}/*/ | cut -d/ -f5`
    for d in $ds; do
      as=`ls -d ${filebase}/${n}/${c}/${d}/*/ | cut -d/ -f6`
      echo $n $c $d $a
      tail -qn1 $filebase/${n}/${c}/${d}/${a}/* > $filebase/${n}_${c}_${d}_${a}.txt;
    done
  done
done
