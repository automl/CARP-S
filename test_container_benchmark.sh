module load system singularity

export OPTIMIZER_CONTAINER=$1
export PROBLEM_CONTAINER=$2

echo "Starting problem container"
singularity instance start "containers/benchmarks/${PROBLEM_CONTAINER}.sif" problem

API_URL="localhost:5000/configspace"  # Replace with the actual API URL

while true; do
    response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

if [ "$response" = "200" ]; then
    echo "API is up and running!"
    break
else
    echo "API is not yet ready (HTTP status: $response). Retrying in 5 seconds..."
    sleep 5
fi
done

echo "Host Found"

echo "Starting optimizer container"
singularity exec "containers/optimizers/${OPTIMIZER_CONTAINER}.sif" python smacbenchmarking/container/container_optimizer.py

echo "Run Finished"