# This includes an upgrade to the new ConfigSpace
# pip install yahpo-gym
git clone https://github.com/benjamc/yahpo_gym.git lib/yahpo_gym
pip install -e lib/yahpo_gym/yahpo_gym
mkdir carps/benchmark_data
cd carps/benchmark_data
git clone https://github.com/slds-lmu/yahpo_data.git
cd ../..
python scripts/patch_yahpo_configspace.py
pip install ConfigSpace --upgrade