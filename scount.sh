ns=`ls data/sing`

for n in $ns; do
  echo $n
  if [ -f data/sing/${n}/scounts.txt ]; then rm data/sing/${n}/scounts.txt; fi
  cs=`ls data/sing/${n}/`
  for c in $cs; do
    echo $n $c
    echo $n $c `tail -n1 -q data/sing/${n}/${c}/* | awk '{n+=$3; t++; if ($4>0){s1++}; if ($5>0){s2++};}END{print(s1/t,s2/t,n/t)}'` >> data/sing/${n}/scounts.txt
  done
done
