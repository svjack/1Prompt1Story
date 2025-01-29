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

huggingface-cli login

# Run gradio demo
python app.py

# Run sample code
python main.py

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

- Anime Background Style Compare
```python
#!huggingface-cli login
from gradio_client import Client
from PIL import Image
from datasets import Dataset
import os
from tqdm import tqdm  # 导入 tqdm

# 配置变量
generated_images_dir = "custom_images_dir"  # 图片保存目录
generated_dataset_dir = "custom_dataset_dir"  # 数据集保存目录
#series_names = ["One Piece", "Naruto", "Attack on Titan"]  # 漫画名称列表
#styles = ["Anime Style", "Realistic Style", "Watercolor Style"]  # 风格选项
series_names = ["One Piece",]  # 漫画名称列表
styles = ["Anime Style",]  # 风格选项

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
    "RunDiffusion/Juggernaut-XI-v11", 
    #"SG161222/RealVisXL_V5.0"
]

# 创建保存图片的目录
os.makedirs(generated_images_dir, exist_ok=True)

# 初始化数据集列表
data = []

# 使用 tqdm 显示总进度条
total_iterations = len(examples) * len(styles) * len(series_names) * len(model_paths)
with tqdm(total=total_iterations, desc="Generating Images", unit="image") as pbar:
    # 遍历每个场景、风格、漫画名称和模型
    for scene_category, scene_prompt in examples.items():
        for style in styles:
            for series_name in series_names:
                for model_path in model_paths:
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

from PIL import Image
ds = dataset.map(lambda x: {"Image": Image.open(x["image"])}).remove_columns(["image"])

ds.push_to_hub("svjack/OnePromptOneStory-AnimeStyle")

print(f"数据集生成并保存完成！图片保存到：{generated_images_dir}，数据集保存到：{generated_dataset_dir}")
```

- Build in Examples
```python
#!huggingface-cli login
from gradio_client import Client
from PIL import Image
from datasets import Dataset
import os
from tqdm import tqdm  # 导入 tqdm 用于进度条

# 配置变量
generated_images_dir = "custom_images_dir"  # 图片保存目录
generated_dataset_dir = "custom_dataset_dir"  # 数据集保存目录

# 初始化 Gradio 客户端
client = Client("http://127.0.0.1:7860")

# 定义 model_path 列表
model_paths = [
    "cagliostrolab/animagine-xl-3.1", 
    "svjack/GenshinImpact_XL_Base",
    "stabilityai/stable-diffusion-xl-base-1.0", 
    "RunDiffusion/Juggernaut-X-v10", 
    "playgroundai/playground-v2.5-1024px-aesthetic", 
    "SG161222/RealVisXL_V4.0", 
    "RunDiffusion/Juggernaut-XI-v11", 
    #"SG161222/RealVisXL_V5.0"
]

# 创建保存图片的目录
os.makedirs(generated_images_dir, exist_ok=True)

# 直接定义 combinations
combinations = [
    {
        "id_prompt": "A hyper-realistic digital painting of a 16 years old girl.",
        "frame_prompt_list": [
            "in a flower garden",
            "building a sandcastle",
            "in a city park with autumn leaves"
        ]
    },
    {
        "id_prompt": "A vintage-style poster of a dog",
        "frame_prompt_list": [
            "playing a guitar at a country concert",
            "sitting by a campfire under a starry sky",
            "riding a skateboard through a bustling city",
            "posing in front of a historical landmark",
            "wearing an astronaut suit on the moon"
        ]
    },
    {
        "id_prompt": "A photo of a dog",
        "frame_prompt_list": [
            "dancing to music at a vibrant street festival",
            "chasing a frisbee in a colorful park",
            "wearing sunglasses while relaxing on a beach chair",
            "posing for a photoshoot in a modern art gallery",
            "jumping through a hoop at a circus performance",
            "playing with a group of children at a playground",
            "exploring a retro diner while wearing a bowtie"
        ]
    },
    {
        "id_prompt": "A mystical illustration of a wise wizard with a long, flowing beard",
        "frame_prompt_list": [
            "in a tower filled with ancient tomes and artifacts",
            "casting a spell by the light of a full moon",
            "standing before a magical portal in the forest",
            "summoning a storm over a mountain peak",
            "writing runes in a dusty spellbook",
            "mixing potions in a dimly lit chamber",
            "consulting a crystal ball"
        ]
    },
    {
        "id_prompt": "A pixar style illustration of a dragon",
        "frame_prompt_list": [
            "soaring gracefully through a rainbow sky",
            "nestled among blooming cherry blossoms",
            "playfully splashing in a sparkling lake"
        ]
    },
    {
        "id_prompt": "A whimsical painting of a delicate fairy",
        "frame_prompt_list": [
            "hovering over a moonlit pond",
            "dancing on the petals of a giant flower",
            "spreading fairy dust over a sleeping village",
            "sitting on a mushroom in a magical forest",
            "playing with fireflies at dusk"
        ]
    },
    {
        "id_prompt": "A hyper-realistic digital painting of an elderly gentleman",
        "frame_prompt_list": [
            "wearing a smoking jacket",
            "at a vintage car show",
            "wearing a vineyard owner's attire",
            "on a golf course",
            "at a classical music concert",
            "painting a landscape"
        ]
    },
    {
        "id_prompt": "A vintage-style poster of a ceramic vase with an intricate floral pattern and a glossy, sky-blue glaze",
        "frame_prompt_list": [
            "holding a rare bouquet of flowers",
            "displaying exotic orchids",
            "complementing a corporate decor",
            "containing delicate cherry blossoms",
            "holding a vibrant arrangement of sunflowers",
            "filled with a fresh bouquet of lavender and wild daisies"
        ]
    },
    {
        "id_prompt": "A photo of a happy hedgehog with its cheese",
        "frame_prompt_list": [
            "in an autumn forest",
            "next to a tiny cheese wheel",
            "sitting on a mushroom",
            "under a picnic blanket",
            "amid blooming spring flowers"
        ]
    },
    {
        "id_prompt": "A heartwarming illustration of a friendly troll",
        "frame_prompt_list": [
            "under a stone bridge covered in ivy",
            "guarding a treasure chest in a dark cave",
            "helping travelers across a river",
            "sitting by a campfire in a foggy forest",
            "building a shelter from fallen logs",
            "fishing in a quiet stream at dusk",
            "carving runes into a rock",
            "resting under a large oak tree"
        ]
    },
    {
        "id_prompt": "A quaint illustration of a hobbit",
        "frame_prompt_list": [
            "in a cozy, round door cottage",
            "sitting by a fireplace in a quaint home",
            "working in a garden of vibrant vegetables",
            "enjoying a feast under a starlit sky",
            "reading a book in a sunlit meadow",
            "walking through a peaceful village",
            "celebrating with friends in a rustic tavern",
            "exploring a hidden valley"
        ]
    },
    {
        "id_prompt": "A hyper-realistic digital painting of a young ginger boy with his ball",
        "frame_prompt_list": [
            "leaves scattering in a gentle breeze",
            "standing in a quiet meadow",
            "set against a vibrant sunset",
            "in a busy street of people",
            "by a colorful graffiti wall",
            "amidst a field of blooming wildflowers"
        ]
    },
    {
        "id_prompt": "A cinematic portrait of a man and a woman standing together",
        "frame_prompt_list": [
            "under a sky full of stars",
            "on a bustling city street at night",
            "in a dimly lit jazz club",
            "walking along a sandy beach at sunset",
            "in a cozy coffee shop with large windows",
            "in a vibrant art gallery surrounded by paintings",
            "under an umbrella during a soft rain",
            "on a quiet park bench amidst falling leaves",
            "standing on a rooftop overlooking the city skyline"
        ]
    },
    {
        "id_prompt": "A cinematic portrait of a man, a woman, and a child",
        "frame_prompt_list": [
            "walking in a quiet park",
            "under a starlit sky",
            "by a rustic cabin",
            "on a forest trail",
            "by a peaceful lake",
            "at a vibrant market",
            "in a snowy street",
            "by a carousel",
            "on a picnic blanket"
        ]
    }
]

# 初始化数据集列表
data = []

# 使用 tqdm 显示总进度条
total_iterations = len(model_paths) * len(combinations)
with tqdm(total=total_iterations, desc="Generating Images", unit="image") as pbar:
    # 遍历每个组合和模型
    for combination in combinations:
        id_prompt = combination["id_prompt"]
        frame_prompt_list = combination["frame_prompt_list"]
        
        for model_path in model_paths:
            # 将 frame_prompt_list 转换为逗号分隔的字符串
            frame_prompt_str = ",".join(frame_prompt_list)
            
            # 调用 Gradio 客户端生成图片
            result = client.predict(
                model_path=model_path,
                id_prompt=id_prompt,
                frame_prompt_list=frame_prompt_str,  # 使用逗号分隔的字符串
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
            image_name = f"{model_path.split('/')[-1]}_{id_prompt.replace(' ', '_')}.png"
            image_path = os.path.join(generated_images_dir, image_name)
            Image.open(result).save(image_path)
            
            # 将数据添加到数据集列表
            data.append({
                "model_name": model_path.split('/')[-1],
                "id_prompt": id_prompt,
                "frame_prompt": frame_prompt_str,  # 保存逗号分隔的字符串
                "image": image_path
            })
            
            # 更新进度条
            pbar.update(1)

# 创建数据集对象
dataset = Dataset.from_dict({
    "model_name": [item["model_name"] for item in data],
    "id_prompt": [item["id_prompt"] for item in data],
    "frame_prompt": [item["frame_prompt"] for item in data],
    "image": [item["image"] for item in data]
})

# 创建保存数据集的目录
os.makedirs(generated_dataset_dir, exist_ok=True)

# 保存数据集到本地
dataset.save_to_disk(generated_dataset_dir)

# 将图像加载到数据集并推送到 Hugging Face Hub
from PIL import Image
ds = dataset.map(lambda x: {"Image": Image.open(x["image"])}).remove_columns(["image"])

# 推送到 Hugging Face Hub
ds.push_to_hub("svjack/OnePromptOneStory-Examples")


print(f"数据集生成并保存完成！图片保存到：{generated_images_dir}，数据集保存到：{generated_dataset_dir}")
```

```python
!pip install -U gradio_client
!pip install datasets

from datasets import load_dataset
from PIL import Image
import io

# 1. 加载数据集
ds = load_dataset("svjack/OnePromptOneStory-Examples")

def bytes_to_image(image_bytes):
    """
    将字节数据（bytes）转换为 PIL.Image 对象。

    参数:
        image_bytes (bytes): 图片的字节数据。

    返回:
        PIL.Image: 转换后的图片对象。
    """
    # 使用 io.BytesIO 将字节数据转换为文件流
    image_stream = io.BytesIO(image_bytes)
    # 使用 PIL.Image.open 打开图片流
    image = Image.open(image_stream)
    return image
    

# 2. 定义图片分割函数
def split_image(image, sub_image_width=None, sub_image_height=None):
    """
    将图片分割成多个子图片。
    
    参数:
        image (PIL.Image): 输入的图片对象。
        sub_image_width (int): 每个子图片的宽度。如果为 None，则不水平分割。
        sub_image_height (int): 每个子图片的高度。如果为 None，则不垂直分割。
    
    返回:
        list: 包含所有子图片的列表。
    """
    # 获取图片的宽度和高度
    width, height = image.size
    
    # 初始化子图片列表
    sub_images = []
    
    # 水平分割
    if sub_image_width is not None:
        # 计算可以分割成多少个子图片
        num_horizontal = width // sub_image_width
        for i in range(num_horizontal):
            left = i * sub_image_width
            right = (i + 1) * sub_image_width
            # 裁剪图片
            sub_image = image.crop((left, 0, right, height))
            sub_images.append(sub_image)
    
    # 垂直分割
    if sub_image_height is not None:
        # 计算可以分割成多少个子图片
        num_vertical = height // sub_image_height
        for j in range(num_vertical):
            top = j * sub_image_height
            bottom = (j + 1) * sub_image_height
            # 裁剪图片
            sub_image = image.crop((0, top, width, bottom))
            sub_images.append(sub_image)
    
    # 如果既没有水平分割也没有垂直分割，返回原图
    if not sub_images:
        sub_images.append(image)
    
    return sub_images

# 3. 定义一个函数来处理每个样本
def process_example(example):
    image = example["Image"]
    # 调用分割函数，假设水平分割宽度为 1024
    example["sub_images"] = split_image(image, sub_image_width=1024)
    return example

# 4. 应用函数到整个数据集
ds = ds.map(process_example, num_proc = 6)

# 5. 查看结果
print(ds["train"][0]["sub_images"])

# 6. 显示第一张子图片（可选）
bytes_to_image(ds["train"][0]["sub_images"][0]["bytes"])

bytes_to_image(ds["train"][0]["sub_images"][0]["bytes"]).save("im.png")
```

![im](https://github.com/user-attachments/assets/8df2c81c-ec65-4440-bf48-a9e100ef195f)


- AnimateLCM-SVD I2V model
```bash
git clone https://huggingface.co/spaces/svjack/AnimateLCM-SVD-Genshin-Impact-Demo && cd AnimateLCM-SVD-Genshin-Impact-Demo && pip install -r requirements.txt
python app.py
```

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
result = client.predict(
		"im.png",	# filepath  in 'Upload your image' Image component
		0,	# float (numeric value between 0 and 9223372036854775807) in 'Seed' Slider component
		True,	# bool  in 'Randomize seed' Checkbox component
		20,	# float (numeric value between 1 and 255) in 'Motion bucket id' Slider component
		8,	# float (numeric value between 5 and 30) in 'Frames per second' Slider component
		1.2,	# float (numeric value between 1 and 2) in 'Max guidance scale' Slider component
		1,	# float (numeric value between 1 and 1.5) in 'Min guidance scale' Slider component
		1024,	# float (numeric value between 576 and 2048) in 'Width of input image' Slider component
		1024,	# float (numeric value between 320 and 1152) in 'Height of input image' Slider component
		4,	# float (numeric value between 1 and 20) in 'Num inference steps' Slider component
		api_name="/video"
)
print(result)

from shutil import copy2
from IPython import display
copy2(result[0]["video"], "vid.mp4")
display.Video("vid.mp4", width = 512, height = 512)
```


https://github.com/user-attachments/assets/ca136c1c-3392-4f00-9d24-fa6390e53575


```python
import uuid
from datasets import load_dataset
from PIL import Image
import io
from gradio_client import Client
import os
import argparse

def bytes_to_image(image_bytes):
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    return image

def split_image(image, sub_image_width=None, sub_image_height=None):
    width, height = image.size
    sub_images = []
    
    if sub_image_width is not None:
        num_horizontal = width // sub_image_width
        for i in range(num_horizontal):
            left = i * sub_image_width
            right = (i + 1) * sub_image_width
            sub_image = image.crop((left, 0, right, height))
            sub_images.append(sub_image)
    
    if sub_image_height is not None:
        num_vertical = height // sub_image_height
        for j in range(num_vertical):
            top = j * sub_image_height
            bottom = (j + 1) * sub_image_height
            sub_image = image.crop((0, top, width, bottom))
            sub_images.append(sub_image)
    
    if not sub_images:
        sub_images.append(image)
    
    return sub_images

def process_example(example):
    image = example["Image"]
    sub_images = split_image(image, sub_image_width=1024)
    
    example["sub_images"] = []
    for sub_image in sub_images:
        image_bytes = io.BytesIO()
        sub_image.save(image_bytes, format="PNG")
        example["sub_images"].append({"bytes": image_bytes.getvalue()})
    
    return example

def generate_video(image_bytes, motion_bucket_id):
    image = bytes_to_image(image_bytes)
    unique_filename = str(uuid.uuid4()) + ".png"
    image.save(unique_filename)
    
    client = Client("http://127.0.0.1:7860")
    result = client.predict(
        unique_filename, 0, True, motion_bucket_id, 8, 1.2, 1, 1024, 1024, 4, api_name="/video"
    )
    
    os.remove(unique_filename)
    
    with open(result[0]["video"], "rb") as video_file:
        video_bytes = video_file.read()
    
    os.remove(result[0]["video"])
    
    return video_bytes

def add_video_to_example(example):
    example["videos"] = []
    for sub_image_dict in example["sub_images"]:
        image_bytes = sub_image_dict["bytes"]
        for motion_bucket_id in [10, 20, 30, 40, 50]:
            video_bytes = generate_video(image_bytes, motion_bucket_id)
            example["videos"].append({"video_bytes": video_bytes, "motion_bucket_id": motion_bucket_id})
    return example

def main(dataset_name, start_index, end_index, output_dir):
    # 加载数据集并选择 "train" 子集
    ds = load_dataset(dataset_name)["train"].select(range(start_index, end_index))
    
    # 处理数据集
    ds = ds.map(process_example, num_proc=6)
    ds = ds.map(add_video_to_example, num_proc=1)
    
    # 保存处理后的数据集
    output_path = os.path.join(output_dir, f"dataset_{start_index}_{end_index}")
    ds.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset and generate videos.")
    parser.add_argument("--dataset_name", type=str, default="svjack/OnePromptOneStory-Examples", 
                        help="Name of the dataset to process (default: svjack/OnePromptOneStory-Examples)")
    parser.add_argument("--start", type=int, required=True, help="Start index of the dataset")
    parser.add_argument("--end", type=int, required=True, help="End index of the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the dataset")
    
    args = parser.parse_args()
    
    main(args.dataset_name, args.start, args.end, args.output_dir)
```

```bash
#!/bin/bash

# 数据集名称
DATASET_NAME="svjack/OnePromptOneStory-Examples"

# 输出目录
OUTPUT_DIR="OnePromptOneStory-Examples"

# 数据集总大小
TOTAL_ITEMS=98

# 每次处理的样本数量
BATCH_SIZE=5

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 循环处理数据集
for ((START=0; START<TOTAL_ITEMS; START+=BATCH_SIZE)); do
    END=$((START + BATCH_SIZE))
    if ((END > TOTAL_ITEMS)); then
        END=$TOTAL_ITEMS
    fi

    echo "Processing items from $START to $((END-1))..."

    # 调用 Python 脚本
    python Exp.py --dataset_name "$DATASET_NAME" --start "$START" --end "$END" --output_dir "$OUTPUT_DIR"

    echo "Finished processing items from $START to $((END-1))."
done

echo "All items processed and saved to $OUTPUT_DIR."
```

```python
from datasets import load_from_disk, concatenate_datasets
import pathlib 
import os
import pandas as pd
import numpy as np
l = sorted(pd.Series(list(pathlib.Path("OnePromptOneStory-Examples/").rglob("dataset_*"))).map(
    lambda x: x if os.path.isdir(x) else np.nan
).dropna().map(str).values.tolist(), key = lambda x: 
       list(map(int ,x.split("_")[-2:]))
      )
ds_l = list(map(load_from_disk, l))
ds = concatenate_datasets(ds_l)
ds.push_to_hub("svjack/OnePromptOneStory-Examples-Vid-head75")

from datasets import Dataset, load_dataset
from PIL import Image
import io

def bytes_to_image(image_bytes):
    """
    将字节数据（bytes）转换为 PIL.Image 对象。
    """
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    return image

# 加载数据集
ds = load_dataset("svjack/OnePromptOneStory-Examples-Vid-head75")["train"]

# 定义 motion_bucket_ids
motion_bucket_ids = [10, 20, 30, 40, 50]

# 创建一个新的数据集列表
new_data = []

# 遍历原始数据集中的每一行
for example in tqdm(ds):
    sub_images = example["sub_images"]
    videos = example["videos"]
    
    # 遍历每个 sub_image
    for idx, sub_image_dict in enumerate(sub_images):
        sub_image_bytes = sub_image_dict["bytes"]
        sub_image = bytes_to_image(sub_image_bytes)
        
        # 计算对应的视频索引
        video_idx = idx * len(motion_bucket_ids)
        
        # 遍历每个 motion_bucket_id 和对应的视频
        for i, motion_bucket_id in enumerate(motion_bucket_ids):
            video_dict = videos[video_idx + i]
            video_bytes = video_dict["video_bytes"]  # 视频的二进制数据
            
            # 创建新的样本，保留原始数据的所有字段
            new_sample = {
                **example,  # 保留原始数据的所有字段
                "sub_image": sub_image,
                "motion_bucket_id": motion_bucket_id,
                "video": video_bytes  # 直接存储视频的二进制数据
            }

            new_sample = dict(
                filter(lambda t2: t2[0] not in ["sub_images", "videos"], new_sample.items())
            )
            # 添加到新数据集中
            new_data.append(new_sample)

print(len(new_data))
# 将新数据转换为 Hugging Face Dataset 对象
new_dataset = Dataset.from_list(new_data)

# 查看新数据集中的第一个样本
#print(new_dataset[0])

new_dataset.push_to_hub("svjack/OnePromptOneStory-Examples-Vid-head75-Exp")
```

```python
from datasets import load_from_disk
from PIL import Image
from IPython import display
import io

def bytes_to_image(image_bytes):
    """
    将字节数据（bytes）转换为 PIL.Image 对象。

    参数:
        image_bytes (bytes): 图片的字节数据。

    返回:
        PIL.Image: 转换后的图片对象。
    """
    # 使用 io.BytesIO 将字节数据转换为文件流
    image_stream = io.BytesIO(image_bytes)
    # 使用 PIL.Image.open 打开图片流
    image = Image.open(image_stream)
    return image

# 加载数据集
ds = load_from_disk("example_video_dataset")
### ds.push_to_hub("svjack/OnePromptOneStory-Examples-Vid-head2")
### ds.push_to_hub("svjack/OnePromptOneStory-Examples-Vid")

### ds = load_from_disk("anime_video_dataset")
### ds.push_to_hub("svjack/OnePromptOneStory-AnimeStyle-Vid")

# 获取第一个样本
example = ds[0]

# 选择第 idx 个子图片和对应的视频
idx = 2  # 假设选择第 2 个子图片

# 获取子图片
sub_image_dict = example["sub_images"][idx]  # 获取第 idx 个子图片的字典
sub_image_bytes = sub_image_dict["bytes"]  # 获取二进制数据
sub_image = bytes_to_image(sub_image_bytes)  # 还原为 PIL.Image 对象
sub_image

# 获取对应的视频
# 假设每个子图片生成 5 个视频（对应不同的 motion_bucket_id）
motion_bucket_ids = [10, 20, 30, 40, 50]  # 假设 motion_bucket_id 的取值
video_idx = idx * len(motion_bucket_ids)  # 计算视频的起始索引

# 遍历该子图片对应的所有视频
for i, motion_bucket_id in enumerate(motion_bucket_ids):
    video_dict = example["videos"][video_idx + i]  # 获取对应的视频字典
    video_bytes = video_dict["video_bytes"]  # 获取二进制数据

    # 将二进制数据保存为 MP4 文件
    def bytes_to_video(video_bytes, output_path="output.mp4"):
        """
        将二进制数据转换为视频文件并保存到指定路径。

        参数:
            video_bytes (bytes): 视频的二进制数据。
            output_path (str): 视频文件的保存路径。

        返回:
            str: 视频文件的路径。
        """
        with open(output_path, "wb") as video_file:
            video_file.write(video_bytes)
        return output_path

    # 保存视频并显示
    video_path = bytes_to_video(video_bytes, f"example_video_{motion_bucket_id}.mp4")  # 保存为 MP4 文件
    print(f"Displaying video for sub_image {idx} with motion_bucket_id {motion_bucket_id}")
    display.Video(video_path, width=512, height=512)  # 在 Jupyter Notebook 中显示视频

display.Video("example_video_30.mp4", width=512, height=512)  # 在 Jupyter Notebook 中显示视频
```

![im1](https://github.com/user-attachments/assets/30ea9f27-7dc9-461e-a55e-1046330fa7dd)



https://github.com/user-attachments/assets/fd41f9b8-3896-4caf-bf62-4b8b449b7271


## Benchmark 

---

### 独立运行命令

1. **`cagliostrolab/animagine-xl-4.0`**
   ```bash
   python -m resource.gen_benchmark \
       --save_dir ./result/benchmark/cagliostrolab_animagine-xl-4.0 \
       --benchmark_path ./resource/consistory+.yaml \
       --device cuda:0 \
       --num_gpus 1 \
       --model_path "cagliostrolab/animagine-xl-4.0"
   ```

2. **`cagliostrolab/animagine-xl-3.1`**
   ```bash
   python -m resource.gen_benchmark \
       --save_dir ./result/benchmark/cagliostrolab_animagine-xl-3.1 \
       --benchmark_path ./resource/consistory+.yaml \
       --device cuda:0 \
       --num_gpus 1 \
       --model_path "cagliostrolab/animagine-xl-3.1"
   ```

3. **`svjack/GenshinImpact_XL_Base`**
   ```bash
   python -m resource.gen_benchmark \
       --save_dir ./result/benchmark/svjack_GenshinImpact_XL_Base \
       --benchmark_path ./resource/consistory+.yaml \
       --device cuda:0 \
       --num_gpus 1 \
       --model_path "svjack/GenshinImpact_XL_Base"
   ```

4. **`stabilityai/stable-diffusion-xl-base-1.0`**
   ```bash
   python -m resource.gen_benchmark \
       --save_dir ./result/benchmark/stabilityai_stable-diffusion-xl-base-1.0 \
       --benchmark_path ./resource/consistory+.yaml \
       --device cuda:0 \
       --num_gpus 1 \
       --model_path "stabilityai/stable-diffusion-xl-base-1.0"
   ```

5. **`RunDiffusion/Juggernaut-X-v10`**
   ```bash
   python -m resource.gen_benchmark \
       --save_dir ./result/benchmark/RunDiffusion_Juggernaut-X-v10 \
       --benchmark_path ./resource/consistory+.yaml \
       --device cuda:0 \
       --num_gpus 1 \
       --model_path "RunDiffusion/Juggernaut-X-v10"
   ```

6. **`playgroundai/playground-v2.5-1024px-aesthetic`**
   ```bash
   python -m resource.gen_benchmark \
       --save_dir ./result/benchmark/playgroundai_playground-v2.5-1024px-aesthetic \
       --benchmark_path ./resource/consistory+.yaml \
       --device cuda:0 \
       --num_gpus 1 \
       --model_path "playgroundai/playground-v2.5-1024px-aesthetic"
   ```

7. **`SG161222/RealVisXL_V4.0`**
   ```bash
   python -m resource.gen_benchmark \
       --save_dir ./result/benchmark/SG161222_RealVisXL_V4.0 \
       --benchmark_path ./resource/consistory+.yaml \
       --device cuda:0 \
       --num_gpus 1 \
       --model_path "SG161222/RealVisXL_V4.0"
   ```

8. **`RunDiffusion/Juggernaut-XI-v11`**
   ```bash
   python -m resource.gen_benchmark \
       --save_dir ./result/benchmark/RunDiffusion_Juggernaut-XI-v11 \
       --benchmark_path ./resource/consistory+.yaml \
       --device cuda:0 \
       --num_gpus 1 \
       --model_path "RunDiffusion/Juggernaut-XI-v11"
   ```

---

### 整合到 Shell 脚本

如果你希望将这些命令整合到一个 Shell 脚本中，可以按照以下方式编写：

```bash
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
```

---

### 使用方法
1. 将上述脚本保存为 `run_models.sh`。
2. 赋予脚本执行权限：
   ```bash
   chmod +x run_models.sh
   ```
3. 运行脚本：
   ```bash
   ./run_models.sh
   ```

---

### 总结
- **独立命令**：每个模型的运行命令都已单独列出，方便单独执行。
- **Shell 脚本**：整合所有命令到一个脚本中，方便一次性运行所有任务。
- **保存路径**：根据模型名称自动生成保存路径，确保结果不会混淆。

```python
from datasets import load_dataset, DatasetDict, Dataset
import os
from PIL import Image

# 定义根目录
root_dir = "result_cp/benchmark/"

# 初始化数据集字典
dataset_dict = {}

# 遍历 model_name 和 animal 文件夹
model_data = {"model_name": [], "label": [], "image_name": [], "image": [], }  # 新增 "image_name" 列

for model_name in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_name)
    if not os.path.isdir(model_path):
        continue

    # 初始化当前 model_name 的数据集

    for animal_folder in os.listdir(model_path):
        animal_path = os.path.join(model_path, animal_folder)
        if not os.path.isdir(animal_path):
            continue

        # 遍历图像文件
        for image_file in os.listdir(animal_path):
            image_path = os.path.join(animal_path, image_file)
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 打开图像
                image = Image.open(image_path)
                model_data["image"].append(image)
                model_data["label"].append(animal_folder)
                model_data["model_name"].append(model_name)
                # 添加图片名称（去掉扩展名）
                image_name = os.path.splitext(image_file)[0]  # 去掉扩展名
                model_data["image_name"].append(image_name)

# 将当前 model_name 的数据转换为 Hugging Face Dataset
dataset_dict["train"] = Dataset.from_dict(model_data)

# 合并所有 model_name 的数据集
full_dataset = DatasetDict(dataset_dict)

# 查看数据集
print(full_dataset)

# full_dataset.push_to_hub("svjack/OnePromptOneStory-animagine-xl-4-0-Animal")
```

- CCIP Ranking
```python
###!git clone https://huggingface.co/spaces/svjack/ccip && cd ccip && pip install -r requirements.txt 
###!python app.py

import os
import uuid
from datasets import load_dataset
from PIL import Image
import io
from gradio_client import Client, handle_file

# 1. 加载数据集
ds = load_dataset("svjack/OnePromptOneStory-Examples")

def bytes_to_image(image_bytes):
    """
    将字节数据（bytes）转换为 PIL.Image 对象。

    参数:
        image_bytes (bytes): 图片的字节数据。

    返回:
        PIL.Image: 转换后的图片对象。
    """
    # 使用 io.BytesIO 将字节数据转换为文件流
    image_stream = io.BytesIO(image_bytes)
    # 使用 PIL.Image.open 打开图片流
    image = Image.open(image_stream)
    return image

# 2. 定义图片分割函数
def split_image(image, sub_image_width=None, sub_image_height=None):
    """
    将图片分割成多个子图片。
    
    参数:
        image (PIL.Image): 输入的图片对象。
        sub_image_width (int): 每个子图片的宽度。如果为 None，则不水平分割。
        sub_image_height (int): 每个子图片的高度。如果为 None，则不垂直分割。
    
    返回:
        list: 包含所有子图片的列表。
    """
    # 获取图片的宽度和高度
    width, height = image.size
    
    # 初始化子图片列表
    sub_images = []
    
    # 水平分割
    if sub_image_width is not None:
        # 计算可以分割成多少个子图片
        num_horizontal = width // sub_image_width
        for i in range(num_horizontal):
            left = i * sub_image_width
            right = (i + 1) * sub_image_width
            # 裁剪图片
            sub_image = image.crop((left, 0, right, height))
            sub_images.append(sub_image)
    
    # 垂直分割
    if sub_image_height is not None:
        # 计算可以分割成多少个子图片
        num_vertical = height // sub_image_height
        for j in range(num_vertical):
            top = j * sub_image_height
            bottom = (j + 1) * sub_image_height
            # 裁剪图片
            sub_image = image.crop((0, top, width, bottom))
            sub_images.append(sub_image)
    
    # 如果既没有水平分割也没有垂直分割，返回原图
    if not sub_images:
        sub_images.append(image)
    
    return sub_images

# 3. 定义一个函数来处理每个样本
def process_example(example):
    image = example["Image"]
    # 调用分割函数，假设水平分割宽度为 1024
    example["sub_images"] = split_image(image, sub_image_width=1024)
    return example

# 4. 应用函数到整个数据集
ds = ds.map(process_example, num_proc=6)

# 5. 初始化 Gradio 客户端
client = Client("http://127.0.0.1:7860")

# 6. 定义一个函数来比较图片并保存结果
def compare_images(example):
    sub_images = example["sub_images"]
    comparison_results = []
    
    # 获取第一个子图片
    first_image = bytes_to_image(sub_images[0]["bytes"])
    first_image_path = f"{uuid.uuid4()}.png"
    first_image.save(first_image_path)
    
    for i in range(1, len(sub_images)):
        # 获取当前子图片
        current_image = bytes_to_image(sub_images[i]["bytes"])
        current_image_path = f"{uuid.uuid4()}.png"
        current_image.save(current_image_path)
        
        # 调用 API 比较图片
        result = client.predict(
            imagex=handle_file(first_image_path),
            imagey=handle_file(current_image_path),
            model_name="ccip-caformer-24-randaug-pruned",
            api_name="/_compare"
        )
        
        # 保存比较结果
        comparison_results.append(str(result))
        
        # 删除临时文件
        os.remove(current_image_path)
    
    # 删除第一个子图片的临时文件
    os.remove(first_image_path)
    
    # 将比较结果保存到数据集中
    example["comparison_results"] = comparison_results
    return example

# 7. 应用比较函数到整个数据集
dss = ds.map(compare_images, num_proc=1)

# 8. 查看结果
print(dss["train"][0]["comparison_results"])

# 9. 保存数据集（可选）
#dss.save_to_disk("example_compare")

dss.map(lambda x: {
    "score_list": list(map(lambda y: eval(y)[0] ,x["comparison_results"])),
    "same_list": list(map(lambda y: eval(y)[1] ,x["comparison_results"]))
}).map(
    lambda x: {
        "max_score": max(x["score_list"]),
        "all_same": all(map(lambda y: "not" not in y.lower(), x["same_list"]))
    }
).remove_columns([
    "comparison_results"
]).sort("max_score").push_to_hub("svjack/OnePromptOneStory-Examples-CCIP")
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
