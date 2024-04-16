# If you want to use yahpo locally and do not want to change to an old ConfigSpace version
# run this :)
pip install yahpo-gym
cd carps
mkdir benchmark_data
cd benchmark_data
git clone https://github.com/slds-lmu/yahpo_data.git
cd ../..
python scripts/patch_yahpo_configspace.py