import cv2
import base64
import time
import os
import json
import openai
import httpx
from datetime import datetime
from openai import OpenAI

# 初始化OpenAI客户端
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def print_progress(current, total, start_time, success_count, error_count):
    """打印美观的进度信息"""
    elapsed = time.time() - start_time
    avg_time = elapsed / (current + 1e-6)
    remaining = avg_time * (total - current)
    
    progress = f"[{current}/{total}]".ljust(10)
    bar = f"{current/total*100:.1f}%".rjust(6)
    time_info = f"Elapsed: {datetime.utcfromtimestamp(elapsed).strftime('%H:%M:%S')} | Remaining: {datetime.utcfromtimestamp(remaining).strftime('%H:%M:%S')}"
    stats = f"Success: {success_count} | Errors: {error_count}"
    
    print(f"\r{progress} {bar} | {time_info} | {stats}", end="", flush=True)

def extract_key_frames(video_path, frame_interval=50):
    """增强版的帧提取函数"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                if buffer is not None:
                    frames.append(base64.b64encode(buffer).decode("utf-8"))
            
            frame_count += 1
        
        return frames if frames else None
    
    except Exception as e:
        print(f"\n视频处理错误: {str(e)}")
        return None
    finally:
        if 'cap' in locals():
            cap.release()

def analyze_video_with_gpt(frames, human_prompt):
    """使用GPT分析视频"""
    system_prompt = """You are a video analyst. Provide sub-event descriptions in JSON format:
    {"descriptions": ["desc1", "desc2"]}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            human_prompt,
            *[{"image": frame, "resize": 768} for frame in frames]
        ]}
    ]

    params = {
        "model": "gpt-4o",
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 500,
        "temperature": 0
    }

    for attempt in range(10):
        try:
            response = client.chat.completions.create(**params)
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"\nGPT请求失败 (尝试 {attempt+1}/10): {str(e)}")
            time.sleep(2 ** attempt)
    
    return {"error": "GPT分析失败"}

def process_video_item(item, base_dir):
    """处理单个视频项"""
    video_path = os.path.join(base_dir, item["data_source"], item["video"])
    
    # 打印当前处理的视频信息
    print(f"\n\n{'='*50}")
    print(f"Processing: {item['id']}")
    print(f"Path: {video_path}")
    
    try:
        # 帧提取
        print("Extracting frames...")
        frames = extract_key_frames(video_path)
        if not frames:
            raise ValueError(f"提取到0帧 (可能视频损坏)")
        
        print(f"Extracted {len(frames)} key frames")
        
        # GPT分析
        human_prompt = item["conversations"][0]["value"].replace("<image>\n", "")
        print("Analyzing with GPT...")
        result = analyze_video_with_gpt(frames, human_prompt)
        
        # 结果处理
        if "descriptions" in result:
            item["gptsplit"] = result["descriptions"]
            status = "SUCCESS"
        else:
            item["gptsplit"] = [f"GPT Error: {result.get('error', 'Unknown')}"]
            status = "GPT_ERROR"
        
        return item, status
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"ERROR: {error_msg}")
        item["gptsplit"] = [f"Processing Error: {error_msg}"]
        return item, "ERROR"

def main(input_json, output_json):
    """主处理函数"""
    print(f"\n{'='*30} Video Processing Start {'='*30}")
    print(f"Input: {input_json}")
    print(f"Output: {output_json}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    total = len(data)
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    for i, item in enumerate(data):
        # 处理当前项
        processed_item, status = process_video_item(
            item, 
            base_dir="/data/P_Yih/huggingface/LLaVA-Video-178K"
        )
        data[i] = processed_item
        
        # 更新统计
        if status == "SUCCESS":
            success_count += 1
        else:
            error_count += 1
        
        # 显示进度
        print_progress(i+1, total, start_time, success_count, error_count)
        
        # 定期保存
        if (i+1) % 5 == 0 or (i+1) == total:
            with open(output_json, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 最终统计
    print(f"\n\n{'='*30} Processing Complete {'='*30}")
    print(f"Total: {total} | Success: {success_count} | Errors: {error_count}")
    print(f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print(f"Output saved to: {output_json}")

if __name__ == "__main__":
    input_json = "/data/P_Yih/CoVS/old.json"
    output_json = "gpt_split_descriptions_new.json"
    main(input_json, output_json)