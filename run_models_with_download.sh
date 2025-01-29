#!/bin/bash

# 定义模型列表（svjack/GenshinImpact_XL_Base 排在 Anime 模型之后）
models=(
    "cagliostrolab/animagine-xl-4.0"
    "cagliostrolab/animagine-xl-3.1"
    "svjack/GenshinImpact_XL_Base"
    "stabilityai/stable-diffusion-xl-base-1.0"
    "RunDiffusion/Juggernaut-X-v10"
    "playgroundai/playground-v2.5-1024px-aesthetic"
    "SG161222/RealVisXL_V4.0"
    "RunDiffusion/Juggernaut-XI-v11"
)

# 定义本地模型存储目录
local_model_dir="./local_models"

# 创建本地模型存储目录
mkdir -p "$local_model_dir"

# 阶段 1：下载所有模型
echo "Starting model download phase..."
for model_path in "${models[@]}"; do
    # 将模型路径转换为本地路径
    local_model_path="$local_model_dir/$(echo $model_path | tr '/' '_')"

    # 如果本地模型目录不存在，则使用 huggingface-cli 下载模型
    if [ ! -d "$local_model_path" ]; then
        echo "Downloading model from Hugging Face: $model_path"
        huggingface-cli download "$model_path" --local-dir "$local_model_path" --local-dir-use-symlinks False
    else
        echo "Model already exists locally: $local_model_path"
    fi
done
echo "Model download phase completed!"
echo "----------------------------------------"

# 阶段 2：运行所有模型任务
echo "Starting model execution phase..."
for model_path in "${models[@]}"; do
    # 根据模型名称生成保存路径
    save_dir="./result/benchmark/$(echo $model_path | tr '/' '_')"
    
    # 将模型路径转换为本地路径
    local_model_path="$local_model_dir/$(echo $model_path | tr '/' '_')"

    echo "Running model: $model_path"
    echo "Saving results to: $save_dir"

    # 运行命令，使用本地模型路径
    python -m resource.gen_benchmark \
        --save_dir "$save_dir" \
        --benchmark_path ./resource/consistory+.yaml \
        --device cuda:0 \
        --num_gpus 1 \
        --model_path "$local_model_path"

    echo "Finished running model: $model_path"
    echo "----------------------------------------"
done

echo "All tasks completed!"
