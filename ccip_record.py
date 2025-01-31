###!git clone https://huggingface.co/spaces/svjack/ccip && cd ccip && pip install -r requirements.txt 
###!python app.py

import os
import uuid
from datasets import load_dataset
from PIL import Image
import io
from gradio_client import Client, handle_file

# 1. 加载数据集
ds = load_dataset("svjack/OnePromptOneStory-animagine-xl-4-0")
ds = ds.filter(lambda x: x["image"].size[0] > 1024)
print(ds)

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
    image = example["image"]
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
    "sub_images" ,"comparison_results"
]).sort("max_score").push_to_hub("svjack/OnePromptOneStory-animagine-xl-4-0-CCIP")


from huggingface_hub import HfApi
from datasets import load_dataset, concatenate_datasets, Dataset

# 初始化 HfApi 客户端
api = HfApi()

# 获取用户 'svjack' 下的所有数据集
datasets = api.list_datasets(author="svjack")

# 过滤出以 "-CCIP" 结尾且不包含 "Examples" 的数据集名称
ccip_datasets = [
    dataset.id for dataset in datasets 
    if dataset.id.endswith("-CCIP") and "Examples" not in dataset.id
]

# 加载所有过滤后的数据集
loaded_datasets = [load_dataset(dataset)["train"] for dataset in ccip_datasets]

# 合并所有数据集
merged_dataset = concatenate_datasets(loaded_datasets)

# 按 "max_score" 排序
sorted_dataset = merged_dataset.sort("max_score")

# 推送到 Hugging Face Hub
sorted_dataset.push_to_hub("svjack/OnePromptOneStory-CCIP-Merged")
