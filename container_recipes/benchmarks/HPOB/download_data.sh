apt install unzip -y
wget https://rewind.tf.uni-freiburg.de/index.php/s/rTwPgaxS2Z7NH39/download/saved-surrogates.zip -P carps/benchmark_data/HPO-B/saved-surrogates
unzip carps/benchmark_data/HPO-B/saved-surrogates/saved-surrogates.zip -d carps/benchmark_data/HPO-B
rm carps/benchmark_data/HPO-B/saved-surrogates/saved-surrogates.zip