#!/bin/bash

# $1: First argument is the run identifier or name

export HF_HOME="/anvme/workspace/b273dd14-swe/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

model_name="gpt-5"
#model_name="Qwen/Qwen3-32B"
log_dir='/anvme/workspace/b273dd14-swe/logs'
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

run_id="${1:-unspecified_run}"
swe_log_file="$log_dir/swe_$(basename "$0")-${TIMESTAMP}-${run_id}.log"

mkdir -p $log_dir


#########################################################
# mini-SWE-agent 配置和运行


# mini-SWE-agent 配置
OUTPUT_DIR="./swebench_gpt_results_${run_id}"
SLICE="0:1"  # 只运行第一个实例进行测试
WORKERS=1
CONFIG_FILE="/anvme/workspace/b273dd14-swe/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml"

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
script -q -c "python src/minisweagent/run/extra/swebench.py \
  --subset verified \
  --split test \
  --slice $SLICE \
  -o $OUTPUT_DIR \
  -w $WORKERS \
  --config $CONFIG_FILE \
  --environment-class singularity \
  --model 'gpt-5' \
  --redo-existing" -f $swe_log_file

SWE_EXIT_CODE=$?

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

