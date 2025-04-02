import os
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict
import copy

def extract_canny_features(frame: np.ndarray) -> np.ndarray:
    """仅使用Canny边缘特征"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # 边缘像素分布直方图
        hist = cv2.calcHist([edges], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, None).flatten()
        return hist
    except Exception as e:
        print(f"Canny特征提取失败: {str(e)}")
        return None

def detect_event_boundaries(video_path: str) -> List[int]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        features = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            feat = extract_canny_features(frame)
            if feat is not None:
                features.append(feat)
        
        if len(features) < 2:
            return []
        
        n_clusters = min(4, max(1, len(features)//30))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(features)
        labels = kmeans.labels_
        
        boundaries = [i for i in range(1, len(labels)) if labels[i] != labels[i-1]]
        min_interval = int(fps)
        final_boundaries = []
        for b in boundaries:
            if not final_boundaries or (b - final_boundaries[-1] >= min_interval):
                final_boundaries.append(b)
        
        cap.release()
        return final_boundaries[:5]

    except Exception as e:
        print(f"事件检测失败: {str(e)}")
        return []

def process_entry(entry: Dict, base_dir: str = "./") -> Dict:
    video_rel_path = os.path.join(entry["data_source"], entry["video"])
    video_full_path = os.path.abspath(os.path.join(base_dir, video_rel_path))
    
    if not os.path.exists(video_full_path):
        print(f"Video missing: {video_full_path}")
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
        print(f"Processing failed: {video_full_path} - {str(e)}")
        return None

def build_dataset(input_json: str, output_json: str, base_dir: str = "./"):
    if not os.path.exists(input_json):
        print(f"Input JSON missing: {input_json}")
        return

    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"JSON load failed: {str(e)}")
        return

    results = []
    for idx, entry in enumerate(dataset):
        processed = process_entry(entry, base_dir)
        if processed:
            results.append(processed)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Results: {len(results)}/{len(dataset)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-d", "--data_root", default="./")
    args = parser.parse_args()
    
    build_dataset(args.input, args.output, args.data_root)