ns="50 100"

for n in $ns; do
  echo $n
  if [ -f data/${n}_counts.txt ]; then rm data/${n}_counts.txt; fi
  cs=`ls data/${n}/`
  for c in $cs; do
    ds=`ls data/${n}/${c}`
    for d in $ds; do
      echo $n $c $d
      echo $n $c `tail -n1 -q data/${n}/${c}/${d}/* | awk '{n+=$3; if ($4>0){s1++}; if ($5>0){s2++}; if($6==0){t++;tr+=$7;tp+=$8};if ($6==1){h++;hr+=$7;hp+=$8}; if($3==2){s++;sr+=$7;sp+=$8}}END{print(s1/t,s2/t,n/t,h/t,s/t,hr,hp,sr,sp)}'` >> data/${n}_counts.txt
    done
  done
done
