pip install yahpo-gym
cd smacbenchmarking
mkdir benchmark_data
cd benchmark_data
git clone https://github.com/slds-lmu/yahpo_data.git
python scripts/patch_yahpo_configspace.py