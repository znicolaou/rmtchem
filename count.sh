if [ $# -ne 1 ]; then
echo "usage ./count.sh filebase"
exit
fi
filebase=$1
ns="50"
cs="2 2.5 3"
ds="5 10 25"
for n in $ns; do
  echo $n
  if [ -f $filebase/${n}_counts.txt ]; then rm $filebase/${n}_counts.txt; fi
  for c in $cs; do
    for d in $ds; do
      echo $n $c $d
      #echo $n $c $d `tail -n1 -q data/${n}/${c}/${d}/* | awk 'BEGIN{n=0;tot=0;t=0;h=0;s=0;s1=0;s2=0;hr=0;hp=0;sr=0;sp=0;tr=0;tp=0}{n+=$3;tot++; if ($4>0){s1++}; if ($5>0){s2++}; if($6==0){t++;tr+=$7;tp+=$8};if ($6==1){h++;hr+=$7;hp+=$8}; if($6==2){s++;sr+=$7;sp+=$8}}END{print(s1/tot,s2/tot,n/tot,t,h,s,tr/t,tp/t,hr/h,hp/h,sr/s,sp/s)}'` >> data/${n}_counts.txt
    done
  done
done

for n in $ns; do 
  for c in $cs; do 
    for d in $ds; do 
      tail -qn1 $filebase/${n}/${c}/${d}/* > $filebase/${n}_${c}_${d}.txt; 
    done; 
  done; 
done
