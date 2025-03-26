import cv2
import base64
import time
import os
import json
import openai
import httpx
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

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
    print(len(base64Frames), "frames processed for", item["id"])

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

    retries = 10
    for attempt in range(retries):
        try:
            result = client.chat.completions.create(**params)
            break
        except (httpx.TimeoutException, openai.OpenAIError) as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"Timeout, retry {attempt+1}/{retries}, waiting {wait_time}s... Error: {e}")
                time.sleep(wait_time)
            else:
                raise Exception(f"Request failed after {retries} attempts: {e}")

    token_used = {
        'completion_tokens': result.usage.completion_tokens,
        'prompt_tokens': result.usage.prompt_tokens,
        'total_tokens': result.usage.total_tokens
    }
    print(f"Processed {item['id']}:")
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

        try:
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
        except Exception as e:
            print(f"Failed to save JSON: {e}")

if __name__ == "__main__":
    json_path = ''
    main(json_path)