if ! command -v unzip &> /dev/null; then
    echo "unzip could not be found, installing it now. If it errors, might need to run as root. (sudo apt ...)"
    apt update && apt install unzip -y
fi

HPOB_TASK_DATA_DIR=$(python -c "from carps.objective_functions.hpo_b import HPOB_TASK_DATA_DIR; print(HPOB_TASK_DATA_DIR)")
HPOB_SURROGATES_DIR=$(python -c "from carps.objective_functions.hpo_b import HPOB_SURROGATES_DIR; print(HPOB_SURROGATES_DIR)")

if [ -z "$(ls -A $HPOB_SURROGATES_DIR)" ]; then
    echo "Directory is empty, proceeding with download."
else
    echo "Directory is not empty, skipping download."
    exit 0
fi
wget https://rewind.tf.uni-freiburg.de/index.php/s/rTwPgaxS2Z7NH39/download/saved-surrogates.zip -P $HPOB_SURROGATES_DIR
unzip $HPOB_SURROGATES_DIR/saved-surrogates.zip -d $HPOB_TASK_DATA_DIR
rm $HPOB_SURROGATES_DIR/saved-surrogates.zip