OPT=$1
echo $OPT

# YAHPO all. 856
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=1 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=2 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=3 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=4 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=5 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=6 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=7 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=8 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=9 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=10 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=11 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=12 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=13 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=14 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=15 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=16 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=17 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=18 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=19 -m
python -m carps.run +cluster=noctua $OPT '+problem/YAHPO/blackbox=glob(*)' seed=20 -m
