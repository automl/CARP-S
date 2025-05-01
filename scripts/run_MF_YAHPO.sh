OPT=$1
echo $OPT

# YAHPO MF all. 1653
# repl 797
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=1 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=2 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=3 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=4 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=5 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=6 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=7 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=8 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=9 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=10 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=11 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=12 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=13 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=14 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=15 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=16 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=17 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=18 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=19 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*repl)' seed=20 -m
# trainsize 817
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=1 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=2 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=3 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=4 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=5 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=6 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=7 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=8 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=9 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=10 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=11 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=12 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=13 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=14 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=15 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=16 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=17 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=18 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=19 -m
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*trainsize)' seed=20 -m
# epoch 39
python -m carps.run +cluster=noctua $OPT '+task/YAHPO/multifidelity/all=glob(*epoch)' 'seed=range(1,21)' -m
