#!/bin/bash

# Get a list of GPU IDs to run (separated by spaces)
gpus="$1"
if [ -z "$gpus" ]; then
    echo "usage: $0 \"GPU_IDS\""
    echo "demo: $0 \"0 2 3\""
    exit 1
fi

# Convert GPU ID strings to arrays
IFS=' ' read -r -a gpu_array <<< "$gpus"

# Start an experiment for each GPU
for i in "${!gpu_array[@]}"; do
    gpu="${gpu_array[$i]}"
    seed=$((100*($i+1)))
    echo "Start the experiment on GPU $gpu with seed value $seed"
    
    XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
    CUDA_VISIBLE_DEVICES=$gpu \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
    python scripts/train_mujoco.py --alg dacer --seed $seed &
    
    # Add a 30-second delay
    if [ $i -lt $((${#gpu_array[@]}-1)) ]; then
        sleep 30
    fi
done

wait