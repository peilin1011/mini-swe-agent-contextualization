#!/bin/bash

# $1: First argument is the run identifier or name

export HF_HOME="/anvme/workspace/b273dd14-swe/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=2,3
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

model_name="Qwen/Qwen3-32B"
log_dir='logs'
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

run_id="${1:-unspecified_run}"
vllm_log_file="$log_dir/vllm_$(basename "$0")-${TIMESTAMP}-${run_id}.log"
swe_log_file="$log_dir/swe_$(basename "$0")-${TIMESTAMP}-${run_id}.log"

mkdir -p $log_dir

#########################################################
# 固定端口为 8000
port=8000

echo "Using fixed port: $port"

# 检查端口是否被占用
if ss -lntu | awk 'NR>1 {print $5}' | sed 's/.*://' | grep -qw "$port"; then
    echo "Error: Port $port is already in use. Please free the port first." >&2
    exit 1
fi

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

💡 Starting mini-SWE-agent evaluation...
🛑 Press Ctrl+C to stop the server

EOF

#########################################################
# mini-SWE-agent 配置和运行

# 设置环境变量
export OPENAI_API_BASE="http://localhost:$port/v1"
export OPENAI_API_KEY="EMPTY"

# mini-SWE-agent 配置
OUTPUT_DIR="./swebench_qwen3_results_${run_id}"
SLICE="0:1"  # 只运行第一个实例进行测试
WORKERS=1
CONFIG_FILE="/anvme/workspace/b273dd14-swe/mini-swe-agent/src/minisweagent/config/extra/swebench_qwen_vllm.yaml"

echo "========================================="
echo "🚀 启动 mini-SWE-agent with Qwen3-32B"
echo "========================================="
echo "📁 输出目录: $OUTPUT_DIR"
echo "🔢 实例切片: $SLICE"
echo "👥 工作线程: $WORKERS"
echo "🌐 API Base: $OPENAI_API_BASE"
echo "📋 配置文件: $CONFIG_FILE"
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo "请确保配置文件已正确保存"
    exit 1
fi

# 运行 mini-SWE-agent
echo "🔍 检查 mini-SWE-agent 目录..."
if [ ! -d "/anvme/workspace/b273dd14-swe/mini-swe-agent" ]; then
    echo "❌ mini-SWE-agent 目录不存在: /anvme/workspace/b273dd14-swe/mini-swe-agent"
    echo "请确保 mini-SWE-agent 已正确安装"
    exit 1
fi

cd /anvme/workspace/b273dd14-swe/mini-swe-agent

echo "🚀 启动 mini-SWE-agent 评估..."
python src/minisweagent/run/extra/swebench.py \
  --subset verified \
  --split test \
  --filter "astropy__astropy-14309" \
  -o $OUTPUT_DIR \
  -w $WORKERS \
  --config $CONFIG_FILE \
  --environment-class singularity \
  --model "Qwen/Qwen3-32B" \
  --redo-existing 2>&1

SWE_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
if [ $SWE_EXIT_CODE -eq 0 ]; then
    echo "🎉 mini-SWE-agent 评估完成！"
else
    echo "❌ mini-SWE-agent 评估失败 (退出码: $SWE_EXIT_CODE)"
fi
echo "========================================="
echo "📁 结果保存在: $OUTPUT_DIR"
echo "📋 SWE-agent 日志: $swe_log_file"
echo "📋 vLLM 日志: $vllm_log_file"
echo "📋 使用的配置文件: $CONFIG_FILE"

# Wait for the process to fully terminate
wait $vllm_pid 2>/dev/null || true