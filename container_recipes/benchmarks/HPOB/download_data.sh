apt install unzip -y
if [ -z "$(ls -A carps/benchmark_data/HPO-B/saved-surrogates)" ]; then
    echo "Directory is empty, proceeding with download."
else
    echo "Directory is not empty, skipping download."
    exit 0
fi
wget https://rewind.tf.uni-freiburg.de/index.php/s/rTwPgaxS2Z7NH39/download/saved-surrogates.zip -P carps/benchmark_data/HPO-B/saved-surrogates
unzip carps/benchmark_data/HPO-B/saved-surrogates/saved-surrogates.zip -d carps/benchmark_data/HPO-B
rm carps/benchmark_data/HPO-B/saved-surrogates/saved-surrogates.zip