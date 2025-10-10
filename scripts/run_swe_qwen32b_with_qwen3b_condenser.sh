#!/bin/bash

# Usage: ./scripts/run_swe_qwen32b_with_qwen3b_condenser.sh [run_id]
# Example: ./scripts/run_swe_qwen32b_with_qwen3b_condenser.sh experiment_001
#
# 此脚本启动两个 vLLM 服务器：
# - 端口 8000: Qwen2.5-Coder-32B-Instruct (主模型，用于代码推理)
# - 端口 8001: Qwen3-32B (轻量模型，用于工作流程压缩)

export HF_HOME="/anvme/workspace/b273dd14-swe/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

main_model="Qwen/Qwen3-32B"
summary_model="Qwen/Qwen3-4B"
log_dir='logs'
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

run_id="${1:-unspecified_run}"
vllm_main_log="$log_dir/vllm_main_${TIMESTAMP}_${run_id}.log"
vllm_summary_log="$log_dir/vllm_summary_${TIMESTAMP}_${run_id}.log"
swe_log_file="$log_dir/swe_dual_${TIMESTAMP}_${run_id}.log"

mkdir -p $log_dir

#########################################################
# 端口配置
main_port=8000
summary_port=8001

echo "Port configuration:"
echo "  Main model $main_model : $main_port"
echo "  Summary model $summary_model : $summary_port"

# 检查端口是否被占用
for port in $main_port $summary_port; do
    if ss -lntu | awk 'NR>1 {print $5}' | sed 's/.*://' | grep -qw "$port"; then
        echo "Error: Port $port is already in use. Please free the port first." >&2
        exit 1
    fi
done

#########################################################

# Clean up vLLM servers if script is interrupted
cleanup() { 
    echo "Script interrupted or exiting. Cleaning up vLLM servers..." >&2
    if [ -n "$main_pid" ] && ps -p "$main_pid" > /dev/null; then
        echo "Stopping main model server (PID: $main_pid)..." >&2
        kill "$main_pid"
        wait "$main_pid" 2>/dev/null 
    fi
    if [ -n "$summary_pid" ] && ps -p "$summary_pid" > /dev/null; then
        echo "Stopping summary model server (PID: $summary_pid)..." >&2
        kill "$summary_pid"
        wait "$summary_pid" 2>/dev/null 
    fi
    echo "All vLLM servers stopped." >&2
}
trap cleanup SIGINT SIGTERM EXIT

#########################################################
# 启动主模型服务器 (Qwen2.5-Coder-32B)

echo ""
echo "🚀 Starting Main Model Server: $main_model"

# 使用 4 GPUs 运行主模型
vllm serve $main_model \
    --tensor_parallel_size 4 \
    --enforce_eager \
    --gpu_memory_utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser deepseek_r1 \
    --rope-scaling '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}' \
    --enable_prefix_caching \
    --max_num_seqs 20 \
    --max_model_len $((128 * 1024 - 8 * 1024)) \
    --seed 41 \
    --port $main_port > $vllm_main_log 2>&1 &

main_pid=$!

echo "Main model server starting (PID: $main_pid, Port: $main_port)"

# 等待主模型初始化
timeout_minutes=9
start_time=$(date +%s)
timeout_seconds=$((timeout_minutes * 60))

echo "Waiting for main model to initialize (timeout: ${timeout_minutes} minutes)..."

while [ $(($(date +%s) - start_time)) -lt $timeout_seconds ]; do
    if ! ps -p $main_pid > /dev/null; then
        echo "❌ Main model server process exited with an error"
        exit 1
    fi
    
    if [ -f "$vllm_main_log" ] && grep -q "Application startup complete." "$vllm_main_log"; then
        echo "✅ Main model initialized successfully"
        break
    fi
    sleep 2
done

if [ $(($(date +%s) - start_time)) -ge $timeout_seconds ]; then
    echo "❌ Main model initialization timed out"
    exit 1
fi

#########################################################
# 启动总结模型服务器 (Qwen3-32B)

echo ""
echo "🚀 Starting Summary Model Server: $summary_model"

# 使用剩余 4 GPUs 运行总结模型
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve $summary_model \
    --tensor_parallel_size 4 \
    --enforce_eager \
    --gpu_memory_utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser deepseek_r1 \
    --rope-scaling '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}' \
    --enable_prefix_caching \
    --max_num_seqs 20 \
    --max_model_len $((64 * 1024)) \
    --seed 42 \
    --port $summary_port > $vllm_summary_log 2>&1 &

summary_pid=$!

echo "Summary model server starting (PID: $summary_pid, Port: $summary_port)"

# 等待总结模型初始化
start_time=$(date +%s)

echo "Waiting for summary model to initialize (timeout: ${timeout_minutes} minutes)..."

while [ $(($(date +%s) - start_time)) -lt $timeout_seconds ]; do
    if ! ps -p $summary_pid > /dev/null; then
        echo "❌ Summary model server process exited with an error"
        exit 1
    fi
    
    if [ -f "$vllm_summary_log" ] && grep -q "Application startup complete." "$vllm_summary_log"; then
        echo "✅ Summary model initialized successfully"
        break
    fi
    sleep 2
done

if [ $(($(date +%s) - start_time)) -ge $timeout_seconds ]; then
    echo "❌ Summary model initialization timed out"
    exit 1
fi

cat <<EOF

======================================================================
🎯 Both vLLM Servers are Ready!
======================================================================
   Main Model ($main_model):
   • PID: $main_pid
   • Port: $main_port
   • API Base: http://localhost:$main_port/v1
   • Log: $vllm_main_log
   
   Summary Model ($summary_model):
   • PID: $summary_pid
   • Port: $summary_port
   • API Base: http://localhost:$summary_port/v1
   • Log: $vllm_summary_log
======================================================================

💡 Starting mini-SWE-agent with dual-model workflow condenser...
🛑 Press Ctrl+C to stop both servers

======================================================================
🚀 mini-SWE-agent: Qwen2.5-Coder-32B + Qwen3-32B Condenser
======================================================================

EOF

#########################################################
# mini-SWE-agent 配置
OUTPUT_DIR="./swebench_qwen32b_qwen3b_results_${run_id}"
SLICE="0:1"
WORKERS=1
CONFIG_FILE="/anvme/workspace/b273dd14-swe/mini-swe-agent/src/minisweagent/config/extra/swebench_qwen32b_qwen3b.yaml"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 运行 mini-SWE-agent
echo "🚀 启动 mini-SWE-agent 评估..."
cd /anvme/workspace/b273dd14-swe/mini-swe-agent

python src/minisweagent/run/extra/swebench_with_condenser.py \
  --subset verified \
  --split test \
  --slice $SLICE \
  -o $OUTPUT_DIR \
  -w $WORKERS \
  --config $CONFIG_FILE \
  --environment singularity \
  --model "Qwen/Qwen3-32B" \
  --redo-existing 

SWE_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================================================="
if [ $SWE_EXIT_CODE -eq 0 ]; then
    echo "🎉 mini-SWE-agent 评估完成！"
else
    echo "❌ mini-SWE-agent 评估失败 (退出码: $SWE_EXIT_CODE)"
fi
echo "======================================================================="
echo "📁 结果目录: $OUTPUT_DIR"
echo "📋 SWE-agent 日志: $swe_log_file"
echo "📋 Main Model 日志: $vllm_main_log"
echo "📋 Summary Model 日志: $vllm_summary_log"
echo "📋 配置文件: $CONFIG_FILE"
echo "======================================================================="

# Wait for both processes to fully terminate
wait $main_pid 2>/dev/null || true
wait $summary_pid 2>/dev/null || true


