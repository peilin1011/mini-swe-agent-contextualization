#!/bin/bash

# $1: First argument is the run identifier or name
# $2: Second argument is an optional user-specified port that vLLM will use

export HF_HOME="/anvme/workspace/b273dd14-swe/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=2,3
model_name="Qwen/Qwen3-32B"
log_dir='logs'
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

run_id="${1:-unspecified_run}"
vllm_log_file="$log_dir/vllm_$(basename "$0")-${TIMESTAMP}-${run_id}.log"

mkdir -p $log_dir

#########################################################

find_free_port() {
    local port_candidate
    local min_port=49152
    local max_port=65535
    local max_attempts=100
    local attempt_num=0

    echo "Attempting to find a free port..." >&2
    while [ "$attempt_num" -lt "$max_attempts" ]; do
        port_candidate=$(shuf -i "${min_port}-${max_port}" -n 1)

        if ! ss -lntu | awk 'NR>1 {print $5}' | sed 's/.*://' | grep -qw "$port_candidate"; then
            echo "Found free port: $port_candidate" >&2
            echo "$port_candidate"
            return 0
        fi
        attempt_num=$((attempt_num + 1))
    done

    echo "Error: Could not find a free port after $max_attempts attempts." >&2
    return 1
}

#########################################################

# Clean up vLLM server if script is interrupted to prevent unneeded GPU usage
cleanup() { 
    echo "Script interrupted or exiting. Cleaning up vLLM server (PID: $vllm_pid)..." >&2
    if [ -n "$vllm_pid" ] && ps -p "$vllm_pid" > /dev/null; then
        kill "$vllm_pid"
        wait "$vllm_pid" 2>/dev/null 
        echo "vLLM server stopped." >&2
    else
        echo "vLLM server (PID: $vllm_pid) not found or already stopped." >&2
    fi
}
trap cleanup SIGINT SIGTERM EXIT

#########################################################
# Determine the port to use
if [ -n "$2" ]; then
    requested_port="$2"
    echo "User requested port: $requested_port"
    if ! ss -lntu | awk 'NR>1 {print $5}' | sed 's/.*://' | grep -qw "$requested_port"; then
        port="$requested_port"
        echo "Requested port $port is free. Using it."
    else
        echo "Requested port $requested_port is occupied. Finding a random available port instead."
        port=$(find_free_port)
        if [ $? -ne 0 ]; then
            exit 1
        fi
    fi
else
    echo "No port specified by user. Finding a random available port."
    port=$(find_free_port)
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

echo "vLLM will use port: $port"

#########################################################

vllm serve $model_name \
    --tensor_parallel_size $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
    --enforce_eager \
    --gpu_memory_utilization 0.95 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser deepseek_r1 \
    --rope-scaling '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}' \
    --enable_prefix_caching \
    --max_num_seqs 23 \
    --max_model_len $((128 * 1024 - 8 * 1024)) \
    --seed 41 \
    --port $port > $vllm_log_file 2>&1 &

vllm_pid=$!

#########################################################

timeout_minutes=9
start_time=$(date +%s)
timeout_seconds=$((timeout_minutes * 60))

echo "Waiting for vLLM server to initialize (timeout: ${timeout_minutes} minutes)..."

while [ $(($(date +%s) - start_time)) -lt $timeout_seconds ]; do
    if ! ps -p $vllm_pid > /dev/null; then
        echo "vLLM server process exited with an error"
        exit 1
    fi
    
    if [ -f "$vllm_log_file" ] && grep -q "Application startup complete." "$vllm_log_file"; then
        echo "vLLM server initialized successfully"
        break
    fi
    sleep 1
done

if [ $(($(date +%s) - start_time)) -ge $timeout_seconds ]; then
    echo "Server initialization timed out after ${timeout_minutes} minutes" 
    kill $vllm_pid
    exit 1
fi

cat <<EOF

🎯 vLLM server is ready (PID: $vllm_pid)!
   • Port: $port
   • Model: $model_name
   • API Base: http://0.0.0.0:$port/v1/
   • Log: $vllm_log_file

💡 You can now run experiments with: python scripts/orchestrate_runs.py
🛑 Press Ctrl+C to stop the server
⏱️  Server will auto-terminate after 1 hour of inactivity

EOF

# Wait for the process to fully terminate
wait $vllm_pid 2>/dev/null || true