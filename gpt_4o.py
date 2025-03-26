import cv2
import base64
import time
import os
import json
import openai
import httpx  # 用于捕获请求超时错误
from openai import OpenAI

# 读取 API Key，避免 NoneType 错误
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 OPENAI_API_KEY")

# 初始化 OpenAI 客户端
client = OpenAI(api_key=api_key)

def process_video(item):
    video_path = os.path.join("../", item["data_source"], item["video"])
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read for", item["id"])

    image_list = list(map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]))

    human_prompt = item["conversations"][0]["value"]
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                human_prompt,
                *image_list,
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 400,
        "temperature": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_p": 1,
    }

    # **添加重试机制**
    retries = 10
    for attempt in range(retries):
        try:
            result = client.chat.completions.create(**params)
            break  # 成功请求后跳出循环
        except (httpx.TimeoutException, openai.OpenAIError) as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # 指数退避策略
                print(f"请求超时，重试 {attempt+1}/{retries}，等待 {wait_time}s... 错误: {e}")
                time.sleep(wait_time)
            else:
                raise Exception(f"请求失败，已重试 {retries} 次: {e}")

    token_used = {
        'completion_tokens': result.usage.completion_tokens,
        'prompt_tokens': result.usage.prompt_tokens,
        'total_tokens': result.usage.total_tokens
    }
    print(f"Processing for {item['id']}:")
    print(token_used)

    response = result.choices[0].message.content
    item["conversations"][1]["value"] = response
    return item

def main(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    total_items = len(json_data)
    batch_size = 5
    for i in range(0, total_items, batch_size):
        batch = json_data[i:i + batch_size]
        for j, item in enumerate(batch):
            current_index = i + j + 1
            print(f"Processing item {current_index}/{total_items} (ID: {item['id']})")
            updated_item = process_video(item)
            batch[j] = updated_item

        json_data[i:i + batch_size] = batch

        # **添加异常处理，避免 JSON 写入失败**
        try:
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
        except Exception as e:
            print(f"JSON 保存失败: {e}")

if __name__ == "__main__":
    json_path = '../0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed_wash_kmeans_gpt.json'
    main(json_path)
