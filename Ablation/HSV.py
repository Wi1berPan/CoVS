import os
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict
import sys
from tqdm import tqdm
import tempfile

def extract_features(frame: np.ndarray) -> np.ndarray:
    """仅使用HSV颜色空间特征"""
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None).flatten()
        return hist
    except Exception as e:
        print(f"\nHSV特征提取失败: {str(e)}", file=sys.stderr)
        return None

def detect_event_boundaries(video_path: str) -> List[int]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        features = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 10 == 0:
                feat = extract_features(frame)
                if feat is not None:
                    features.append(feat)
        
        if len(features) < 2:
            return []
        
        n_clusters = min(4, max(1, len(features)//30))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(features)
        labels = kmeans.labels_
        
        boundaries = [i*10 for i in range(1, len(labels)) if labels[i] != labels[i-1]]
        min_interval = int(fps)
        final_boundaries = []
        for b in boundaries:
            if not final_boundaries or (b - final_boundaries[-1] >= min_interval):
                final_boundaries.append(b)
        
        return final_boundaries[:5]

    except Exception as e:
        print(f"\n事件检测失败: {str(e)}", file=sys.stderr)
        return []
    finally:
        if 'cap' in locals():
            cap.release()

def process_entry(entry: Dict, base_dir: str = "./") -> Dict:
    video_rel_path = os.path.join(entry["data_source"], entry["video"])
    video_full_path = os.path.abspath(os.path.join(base_dir, video_rel_path))
    
    if not os.path.exists(video_full_path):
        print(f"\nVideo missing: {video_full_path}", file=sys.stderr)
        return None

    try:
        boundaries = detect_event_boundaries(video_full_path)
        video_id = os.path.splitext(os.path.basename(entry["video"]))[0]
        
        return {
            "id": video_id,
            "number_sub_event": len(boundaries) + 1,
            "Key frames for event segmentation": boundaries
        }
    except Exception as e:
        print(f"\nProcessing failed: {video_full_path} - {str(e)}", file=sys.stderr)
        return None

def build_dataset(input_json: str, output_json: str, base_dir: str = "./"):
    if not os.path.exists(input_json):
        print(f"\nInput JSON missing: {input_json}", file=sys.stderr)
        return

    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"\nJSON load failed: {str(e)}", file=sys.stderr)
        return

    # 使用临时文件避免写入中断损坏原文件
    temp_path = output_json + '.tmp'
    results = []
    success_count = 0
    save_interval = 10  # 每处理10条保存一次
    
    try:
        with tqdm(dataset, desc="Processing videos", unit="video") as pbar:
            for i, entry in enumerate(pbar):
                processed = process_entry(entry, base_dir)
                if processed:
                    results.append(processed)
                    success_count += 1
                
                # 每处理10条或最后一条时保存
                if (i+1) % save_interval == 0 or (i+1) == len(dataset):
                    # 先写入临时文件
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    # 确保写入完成后再重命名
                    if os.path.exists(output_json):
                        os.remove(output_json)
                    os.rename(temp_path, output_json)
                    
                    pbar.set_postfix({
                        'success': success_count,
                        'saved': len(results)
                    })
    
        print(f"\n处理完成: 成功 {success_count}/{len(dataset)}")
        print(f"结果已保存到: {os.path.abspath(output_json)}")
        
    except Exception as e:
        print(f"\n处理过程中出错: {str(e)}", file=sys.stderr)
        # 如果临时文件存在，尝试恢复
        if os.path.exists(temp_path):
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    recovered = json.load(f)
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(recovered, f, indent=2, ensure_ascii=False)
                print(f"已恢复{len(recovered)}条已处理数据到{output_json}")
            except Exception as e2:
                print(f"恢复失败: {str(e2)}", file=sys.stderr)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="视频事件分割处理工具")
    parser.add_argument("-i", "--input", required=True, help="输入JSON文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出JSON文件路径")
    parser.add_argument("-d", "--data_root", default="./", help="视频文件根目录")
    args = parser.parse_args()
    
    build_dataset(args.input, args.output, args.data_root)