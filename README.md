<h1 align="center">
  <!-- <br>
  <a href="http://www.amitmerchant.com/electron-markdownify"><img src="https://raw.githubusercontent.com/amitmerchant1990/electron-markdownify/master/app/img/markdownify.png" alt="Markdownify" width="200"></a>
  <br> -->
  🔥(ICLR 2025) One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt
  <br>
</h1>


<div align="center">

<a href="https://c5e9af216625826dc6.gradio.live" style="display: inline-block;">
    <img src="./resource/gradio.svg" alt="demo" style="height: 20px; vertical-align: middle;">
</a>&nbsp;
<a href="https://arxiv.org/abs/2501.13554" style="display: inline-block;">
    <img src="https://img.shields.io/badge/arXiv%20paper-2406.06525-b31b1b.svg" alt="arXiv" style="height: 20px; vertical-align: middle;">
</a>&nbsp;
<a href="https://byliutao.github.io/1Prompt1Story.github.io/" style="display: inline-block;">
    <img src="https://img.shields.io/badge/Project_page-More_visualizations-green" alt="project page" style="height: 20px; vertical-align: middle;">
</a>&nbsp;

</div>


<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#license">License</a> •
  <a href="#Citation">Citation</a> 
</p>

<p align="center">
  <img src="./resource/photo.gif" alt="screenshot" />
</p>


## Key Features

* Consistent Identity Image Generation.
* Gradio Demo.
* Consistory+ Benchmark: contains 200 prompt sets, with each set containing between 5 and 10 prompts, categorized into 8 superclasses: humans, animals, fantasy, inanimate, fairy tales, nature, technology.
* Benchmark Generation Code.

```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm

# Clone this repository
git clone https://github.com/svjack/1Prompt1Story

# Go into the repository
cd 1Prompt1Story

### Install dependencies ###
conda create --name 1p1s python=3.10
conda activate 1p1s

# Install ipykernel and add the environment to Jupyter
pip install ipykernel
python -m ipykernel install --user --name 1p1s --display-name "1p1s"
### Install dependencies ENDs ###

# Install PyTorch with CUDA 12.1
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision torchaudio

# Install other dependencies using pip
pip install transformers diffusers opencv-python scipy gradio==4.44.1 sympy==1.13.1
pip install "httpx[socks]" datasets

# Run sample code
python main.py

# Run gradio demo
python app.py

# Run Consistory+ benchmark
python -m resource.gen_benchmark --save_dir ./result/benchmark --benchmark_path ./resource/consistory+.yaml
```

- One Piece Background
```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
result = client.predict(
		model_path="cagliostrolab/animagine-xl-3.1",
		id_prompt="A quaint illustration of the environment and background in One Piece",
		frame_prompt_list="in a flower garden, building with rose, birds in a city park",
		precision="fp16",
		seed=32,
		window_length=10,
		alpha_weaken=0.01,
		beta_weaken=0.05,
		alpha_enhance=-0.01,
		beta_enhance=1,
		ipca_drop_out=0,
		use_freeu="false",
		use_same_init_noise="true",
		api_name="/main_gradio"
)
print(result)

from PIL import Image
from IPython import display

Image.open(result).save("one_piece_flower.png")
display.Image("one_piece_flower.png")
```

![one_piece_flower](https://github.com/user-attachments/assets/a7d9e136-d388-48af-a829-08ae0ecd01bd)

- Anime Style Compare
```python
from gradio_client import Client
from PIL import Image
from datasets import Dataset
import os
from tqdm import tqdm  # 导入 tqdm

# 配置变量
generated_images_dir = "custom_images_dir"  # 图片保存目录
generated_dataset_dir = "custom_dataset_dir"  # 数据集保存目录
series_names = ["One Piece", "Naruto", "Attack on Titan"]  # 漫画名称列表
styles = ["Anime Style", "Realistic Style", "Watercolor Style"]  # 风格选项

# 初始化 Gradio 客户端
client = Client("http://127.0.0.1:7860")

# 定义 examples 和 model_path 列表
examples = {
    "Nature Scene": "Sunlight spills over the emerald grass, a gentle breeze sways blooming flowers, distant mountains fade in and out of view, and a stream whispers softly as it flows.",
    "Sci-Fi Scene": "The spaceship glides through the stars, its blue exhaust cutting through the darkness, robotic arms work tirelessly outside, and an AI voice echoes, 'Destination planet approaching.'",
    "City Nightscape": "Neon lights illuminate the streets, glass skyscrapers reflect the shimmering glow, crowds bustle through the sidewalks, and cars weave endlessly through the noise.",
    "Fantasy Adventure": "In an ancient forest where trees touch the sky, a lone traveler treads softly on moss-covered ground, magical creatures peek from behind the leaves, and a glowing portal hums with untold secrets.",
    "Historical Setting": "The cobblestone streets of the old town echo with history, horse-drawn carriages clatter past stone buildings, and market vendors call out to passersby, their stalls filled with spices, fabrics, and trinkets.",
    "Post-Apocalyptic World": "The ruins of a once-great city stretch to the horizon, broken skyscrapers loom like skeletal giants, dust swirls in the air carried by a dry wind, and a lone figure scavenges for supplies in the silence.",
    "Beach Getaway": "Waves crash against the golden shore, seagulls cry as they glide through the salty air, palm trees sway in the tropical breeze, and the sun sets, painting the sky in hues of orange and pink.",
    "Mystery Scene": "A dimly lit room holds secrets untold, bookshelves line the walls filled with dusty tomes, a single candle flickers on the desk, and long shadows dance with every movement.",
    "Space Exploration": "The astronaut floats in the vast emptiness of space, Earth a distant blue marble in the background, stars twinkle like diamonds in the void, and the silence is both peaceful and overwhelming.",
    "Medieval Battle": "The battlefield roars with the clash of steel, knights in armor charge under fluttering banners, arrows rain down from the sky, and the ground trembles under the weight of marching armies."
}

model_paths = [
    "cagliostrolab/animagine-xl-3.1", "svjack/GenshinImpact_XL_Base",
    "stabilityai/stable-diffusion-xl-base-1.0", "RunDiffusion/Juggernaut-X-v10", 
    "playgroundai/playground-v2.5-1024px-aesthetic", "SG161222/RealVisXL_V4.0", 
    "RunDiffusion/Juggernaut-XI-v11", "SG161222/RealVisXL_V5.0"
]

# 创建保存图片的目录
os.makedirs(generated_images_dir, exist_ok=True)

# 初始化数据集列表
data = []

# 使用 tqdm 显示总进度条
total_iterations = len(model_paths) * len(examples) * len(styles) * len(series_names)
with tqdm(total=total_iterations, desc="Generating Images", unit="image") as pbar:
    # 遍历每个模型、场景、风格和漫画名称
    for model_path in model_paths:
        for scene_category, scene_prompt in examples.items():
            for style in styles:
                for series_name in series_names:
                    # 动态生成 id_prompt，替换模板变量
                    id_prompt = f"A quaint illustration of the environment and background in {series_name}, {style}"
                    
                    # 调用 Gradio 客户端生成图片
                    result = client.predict(
                        model_path=model_path,
                        id_prompt=id_prompt,
                        frame_prompt_list=scene_prompt,
                        precision="fp16",
                        seed=32,
                        window_length=10,
                        alpha_weaken=0.01,
                        beta_weaken=0.05,
                        alpha_enhance=-0.01,
                        beta_enhance=1,
                        ipca_drop_out=0,
                        use_freeu="false",
                        use_same_init_noise="true",
                        api_name="/main_gradio"
                    )
                    
                    # 保存图片到本地
                    image_name = f"{model_path.split('/')[-1]}_{scene_category.replace(' ', '_')}_{style.replace(' ', '_')}_{series_name.replace(' ', '_')}.png"
                    image_path = os.path.join(generated_images_dir, image_name)
                    Image.open(result).save(image_path)
                    
                    # 将数据添加到数据集列表
                    data.append({
                        "model_name": model_path.split('/')[-1],
                        "scene_category": scene_category,
                        "scene_prompt": scene_prompt,
                        "style": style,  # 风格列
                        "series_name": series_name,  # 漫画名称列
                        "image": image_path
                    })
                    
                    # 更新进度条
                    pbar.update(1)

# 创建数据集对象
dataset = Dataset.from_dict({
    "model_name": [item["model_name"] for item in data],
    "scene_category": [item["scene_category"] for item in data],
    "scene_prompt": [item["scene_prompt"] for item in data],
    "style": [item["style"] for item in data],  # 风格列
    "series_name": [item["series_name"] for item in data],  # 漫画名称列
    "image": [item["image"] for item in data]
})

# 创建保存数据集的目录
os.makedirs(generated_dataset_dir, exist_ok=True)

# 保存数据集到本地
dataset.save_to_disk(generated_dataset_dir)

print(f"数据集生成并保存完成！图片保存到：{generated_images_dir}，数据集保存到：{generated_dataset_dir}")

```

## How To Use

```bash
# Clone this repository
$ git clone https://github.com/svjack/1Prompt1Story

# Go into the repository
$ cd 1Prompt1Story

### Install dependencies ###
$ conda create --name 1p1s python=3.10
$ conda activate 1p1s
# choose the right cuda version of your device
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
$ conda install conda-forge::transformers 
$ conda install -c conda-forge diffusers
$ pip install opencv-python scipy gradio=4.44.1 sympy==1.13.1
### Install dependencies ENDs ###

# Run sample code
$ python main.py

# Run gradio demo
$ python app.py

# Run Consistory+ benchmark
$ python -m resource.gen_benchmark --save_dir ./result/benchmark --benchmark_path ./resource/consistory+.yaml
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@inproceedings{
liu2025onepromptonestory,
title={One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt},
author={Tao Liu and Kai Wang and Senmao Li and Joost van de Weijer and Fhad Khan and Shiqi Yang and Yaxing Wang and Jian Yang and Mingming Cheng},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=cD1kl2QKv1}
}
```
