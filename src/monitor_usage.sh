#!/bin/bash

# Directory to store the logs
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Log files
CPU_LOG="$LOG_DIR/cpu_usage.log"
MEMORY_LOG="$LOG_DIR/memory_usage.log"
GPU_LOG="$LOG_DIR/gpu_usage.log"
GPU_MEMORY_LOG="$LOG_DIR/gpu_memory_usage.log"

# Duration to monitor (in seconds)
DURATION=400
INTERVAL=1

# Get container ID of the running Docker container
CONTAINER_ID=$(docker ps -qf "name=my_test_rag_container")

# Function to monitor CPU and memory usage
monitor_cpu_memory() {
    for ((i=0; i<$DURATION; i+=$INTERVAL)); do
        timestamp=$(date +%s)
        cpu_usage=$(docker stats --no-stream --format "{{.CPUPerc}}" $CONTAINER_ID | tr -d '%')
        memory_usage=$(docker stats --no-stream --format "{{.MemUsage}}" $CONTAINER_ID | awk '{print $1}' | tr -d 'MiB')
        echo "$timestamp $cpu_usage" >> $CPU_LOG
        echo "$timestamp $memory_usage" >> $MEMORY_LOG
        sleep $INTERVAL
    done
}

# Function to monitor GPU usage
monitor_gpu() {
    for ((i=0; i<$DURATION; i+=$INTERVAL)); do
        timestamp=$(date +%s)
        if command -v nvidia-smi &> /dev/null; then
            gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
            gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
            echo "$timestamp $gpu_usage" >> $GPU_LOG
            echo "$timestamp $gpu_memory" >> $GPU_MEMORY_LOG
        else
            echo "$timestamp 0" >> $GPU_LOG
            echo "$timestamp 0" >> $GPU_MEMORY_LOG
        fi
        sleep $INTERVAL
    done
}

# Clear previous logs
> $CPU_LOG
> $MEMORY_LOG
> $GPU_LOG
> $GPU_MEMORY_LOG

# Run the monitoring functions in the background
monitor_cpu_memory &
monitor_gpu &

# Wait for monitoring to complete
wait

# Generate graphs using Python
python3 generate_graphs.py $CPU_LOG $MEMORY_LOG $GPU_LOG $GPU_MEMORY_LOG