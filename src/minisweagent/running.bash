salloc --gres=gpu:a40:1 --time=1:00:00
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
module load python/3.12-conda
conda activate mini-swe-agent

python src/minisweagent/run/extra/swebench_single.py -i 0 -m claude-sonnet-4-20250514 --environment-class singularity


python src/minisweagent/run/extra/swebench.py --subset verified --split test --slice "0:1" -o ./swebench_verified_results -w 1 --environment-class singularity -m gpt-5 --redo-existing


export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"

huggingface-cli download Qwen/Qwen3-Coder-32B-Instruct