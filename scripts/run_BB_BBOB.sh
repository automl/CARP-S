
OPT=$1
echo $OPT

python -m carps.run +cluster=noctua $OPT '+problem/BBOB=glob("*")' 'seed=range(1,3)' -m
python -m carps.run +cluster=noctua $OPT '+problem/BBOB=glob("*")' 'seed=range(3,5)' -m
python -m carps.run +cluster=noctua $OPT '+problem/BBOB=glob("*")' 'seed=range(5,7)' -m
python -m carps.run +cluster=noctua $OPT '+problem/BBOB=glob("*")' 'seed=range(7,9)' -m
python -m carps.run +cluster=noctua $OPT '+problem/BBOB=glob("*")' 'seed=range(9,11)' -m
python -m carps.run +cluster=noctua $OPT '+problem/BBOB=glob("*")' 'seed=range(11,13)' -m
python -m carps.run +cluster=noctua $OPT '+problem/BBOB=glob("*")' 'seed=range(13,15)' -m
python -m carps.run +cluster=noctua $OPT '+problem/BBOB=glob("*")' 'seed=range(15,17)' -m
python -m carps.run +cluster=noctua $OPT '+problem/BBOB=glob("*")' 'seed=range(17,19)' -m
python -m carps.run +cluster=noctua $OPT '+problem/BBOB=glob("*")' 'seed=range(19,21)' -m

