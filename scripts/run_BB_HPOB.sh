
# OPT='+optimizer/randomsearch=config'

OPT=$1
echo $OPT
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_2*")' 'seed=range(1,21)' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_3*")' 'seed=range(1,21)' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_4*")' 'seed=range(1,21)' -m
# 900 tasks
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=1' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=2' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=3' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=4' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=5' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=6' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=7' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=8' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=9' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=10' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=11' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=12' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=13' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=14' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=15' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=16' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=17' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=18' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=19' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_5*")' 'seed=20' -m
# 600 tasks
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=1' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=2' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=3' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=4' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=5' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=6' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=7' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=8' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=9' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=10' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=11' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=12' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=13' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=14' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=15' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=16' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=17' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=18' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=19' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_6*")' 'seed=20' -m
# 259
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_7*")' 'seed=range(1,4)' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_7*")' 'seed=range(4,7)' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_7*")' 'seed=range(7,11)' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_7*")' 'seed=range(11,14)' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_7*")' 'seed=range(14,17)' -m
python -m carps.run +cluster=noctua $OPT '+task/HPOB/all=glob("cfg_7*")' 'seed=range(17,21)' -m
