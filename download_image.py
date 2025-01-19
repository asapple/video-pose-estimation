import os
import requests
import time
from datetime import datetime, timedelta
from shutil import copyfile
import argparse

# 配置
API_URL = "http://localhost:8080/stickman/background/getImg"  # 后端接口URL
CACHE_DIR = "image"  # 本地缓存目录
CACHE_EXPIRY_TIME = timedelta(days=1)  # 缓存有效期为1天
DEFAULT_BLANK_IMAGE = "default_blank.jpg"  # 本地默认空白图片路径

# 检查缓存是否过期
def is_cache_valid(device_id):
    cache_file = os.path.join(CACHE_DIR, f"{device_id}.jpg")
    if not os.path.exists(cache_file):
        return False
    file_mod_time = os.path.getmtime(cache_file)
    file_mod_time = datetime.fromtimestamp(file_mod_time)
    return datetime.now() - file_mod_time < CACHE_EXPIRY_TIME

# 获取背景图URL
def get_background_image_url(device_id):
    params = {'device_id': device_id}
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if response.status_code == 200:
            return data.get("data")
        else:
            print(f"Error: {data.get('message')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# 下载图片并缓存
def download_and_cache_image(url, device_id):
    cache_file = os.path.join(CACHE_DIR, f"{device_id}.jpg")
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(cache_file, 'wb') as f:
            f.write(response.content)
        print(f"Image downloaded and cached at {cache_file}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {e}")

# 使用默认空白图片进行缓存
def use_default_blank_image(device_id):
    cache_file = os.path.join(CACHE_DIR, f"{device_id}.jpg")
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    try:
        copyfile(DEFAULT_BLANK_IMAGE, cache_file)
        print(f"Using default blank image, cached at {cache_file}")
    except Exception as e:
        print(f"Failed to copy default image: {e}")

# 获取或下载背景图
def get_or_download_image(device_id):
    if is_cache_valid(device_id):
        print(f"Using cached image at {os.path.join(CACHE_DIR, f'{device_id}.jpg')}")
        return os.path.join(CACHE_DIR, f"{device_id}.jpg")
    else:
        print("Cache is expired or not found. Downloading image...")
        image_url = get_background_image_url(device_id)
        if image_url:
            download_and_cache_image(image_url, device_id)
            return os.path.join(CACHE_DIR, f"{device_id}.jpg")
        else:
            print("Failed to get image URL. Using default blank image.")
            use_default_blank_image(device_id)
            return os.path.join(CACHE_DIR, f"{device_id}.jpg")

# 主程序
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Download background image for device.")
    parser.add_argument('--device_id', type=str, required=True, help="Device ID for background image")
    args = parser.parse_args()

    # 获取或下载背景图
    image_path = get_or_download_image(args.device_id)
    if image_path:
        print(f"Image is available at: {image_path}")
    else:
        print("Could not retrieve the image.")
