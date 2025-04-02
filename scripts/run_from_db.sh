
for i in {1..5}; do
    echo "Running iteration $i"
    python -m carps.run_from_db 'job_nr_dummy=range(1,1000)' -m  
done

 