#!/bin/bash -e
ks=$1

IFS=',' read -r -a ks <<< "$1"

# which subset (dev or test), k, discrepancy

regex='^.*k=([0-9]+).*discrepancy=(0\.[0-9]+).*$'
echo which,k,discrepancy
for value in "${ks[@]}"
do
    if [[ $(head -n 1 subset_$value.txt) =~ $regex ]]
    then 
        echo "s1,${BASH_REMATCH[1]},${BASH_REMATCH[2]}"
    fi
    if [[ $(head -n 1 subset_complement_subset_$value.txt) =~ $regex ]]
    then
        echo "s2,${BASH_REMATCH[1]},${BASH_REMATCH[2]}"
    fi
done