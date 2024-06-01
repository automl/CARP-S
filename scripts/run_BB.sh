OPT=$1
echo $OPT
bash scripts/run_BB_YAHPO.sh $OPT
bash scripts/run_BB_HPOB.sh $OPT
bash scripts/run_BB_HPOBench.sh $OPT
bash scripts/run_BB_BBOB.sh $OPT