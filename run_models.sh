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

# 遍历模型列表并运行任务
for model_path in "${models[@]}"; do
    # 根据模型名称生成保存路径
    save_dir="./result/benchmark/$(echo $model_path | tr '/' '_')"
    
    echo "Running model: $model_path"
    echo "Saving results to: $save_dir"

    # 运行命令
    python -m resource.gen_benchmark \
        --save_dir "$save_dir" \
        --benchmark_path ./resource/consistory+.yaml \
        --device cuda:0 \
        --num_gpus 1 \
        --model_path "$model_path"

    echo "Finished running model: $model_path"
    echo "----------------------------------------"
done

echo "All tasks completed!"
