apt install unzip -y
wget https://rewind.tf.uni-freiburg.de/index.php/s/rTwPgaxS2Z7NH39/download/saved-surrogates.zip -P smacbenchmarking/benchmark_data/HPO-B/saved-surrogates
unzip smacbenchmarking/benchmark_data/HPO-B/saved-surrogates/saved-surrogates.zip -d smacbenchmarking/benchmark_data/HPO-B
rm smacbenchmarking/benchmark_data/HPO-B/saved-surrogates/saved-surrogates.zip