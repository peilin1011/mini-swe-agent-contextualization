#!/bin/bash

# Usage: ./scripts/run_swe_qwen32b_with_qwen3b_condenser.sh [run_id]
# Example: ./scripts/run_swe_qwen32b_with_qwen3b_condenser.sh experiment_001
#
# 此脚本启动两个 vLLM 服务器：
# - 端口 8000: Qwen2.5-Coder-32B-Instruct (主模型，用于代码推理)
# - 端口 8001: Qwen3-32B (轻量模型，用于工作流程压缩)

run_id="${1:-unspecified_run}"

main_model="gpt-5"
summary_model="gpt-5"
log_dir='logs'
TIMESTAMP=$(date +%Y%m%d-%H%M%S)


#########################################################
# mini-SWE-agent 配置
OUTPUT_DIR="./swebench_local_gpt5_results_${run_id}"
SLICE="0:10"
WORKERS=1
CONFIG_FILE="src/minisweagent/config/extra/swebench_local_gpt5.yaml"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 运行 mini-SWE-agent
echo "🚀 启动 mini-SWE-agent 评估..."

python src/minisweagent/run/extra/swebench_with_condenser.py \
  --subset verified \
  --split test \
  --slice $SLICE \
  -o $OUTPUT_DIR \
  -w $WORKERS \
  --config $CONFIG_FILE \
  --environment docker \
  --model "gpt-5" \
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


