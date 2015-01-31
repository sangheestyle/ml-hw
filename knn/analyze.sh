#!/bin/bash
# print 'k, n, accuracy'
# usage: $ bash analyze.sh > result.csv

k=(1 3 5 7 9)
n=(100 200 300 400 500)

printf "k,n,accuracy\n"
for kk in ${k[*]}
do
  for nn in ${n[*]}
  do
    output=$(python knn.py --k $kk --limit $nn | grep Accuracy)
    accuracy=${output/Accuracy: /}
    printf "%d,%d,%f\n" "$kk" "$nn" "$accuracy"
  done
done
