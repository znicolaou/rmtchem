ns="50 100 200"

for n in $ns; do
  echo $n
  if [ -f data/${n}_counts.txt ]; then rm data/${n}_counts.txt; fi
  cs=`ls data/${n}/`
  for c in $cs; do
    ds=`ls data/${n}/${c}`
    for d in $ds; do
      echo $n $c $d
      echo $n $c $d `tail -n1 -q data/${n}/${c}/${d}/* | awk 'BEGIN{n=0;tot=0;t=0;h=0;s=0;s1=0;s2=0;hr=0;hp=0;sr=0;sp=0;tr=0;tp=0}{n+=$3;tot++; if ($4>0){s1++}; if ($5>0){s2++}; if($6==0){t++;tr+=$7;tp+=$8};if ($6==1){h++;hr+=$7;hp+=$8}; if($6==2){s++;sr+=$7;sp+=$8}}END{print(s1/tot,s2/tot,n/tot,t,h,s,tr/t,tp/t,hr/h,hp/h,sr/s,sp/s)}'` >> data/${n}_counts.txt
    done
  done
done
