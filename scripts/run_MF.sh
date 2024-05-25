OPT=$1
echo $OPT

bash scripts/run_MF_HPOBench.sh $OPT
bash scripts/run_MF_YAHPO.sh $OPT
bash scripts/run_MF_MFPBench.sh $OPT