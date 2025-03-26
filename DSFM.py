import os
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List, Dict
from collections import defaultdict

def extract_features(frame: np.ndarray) -> np.ndarray:
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        return np.hstack([color_hist, edge_hist])
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        return None

def compute_frame_difference(prev: np.ndarray, curr: np.ndarray) -> float:
    try:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        return np.sum(diff) / (diff.size * 255)
    except Exception as e:
        print(f"Frame difference failed: {str(e)}")
        return 0.0

def compute_motion_histogram(prev: np.ndarray, curr: np.ndarray) -> float:
    try:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
        if p0 is None:
            return 0.0
        p1, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        motion_vectors = np.linalg.norm(p1 - p0, axis=2)
        motion_hist = np.histogram(motion_vectors, bins=10, range=(0, 10))[0]
        return np.sum(motion_hist) / (len(p0) * 10)
    except Exception as e:
        print(f"Motion detection failed: {str(e)}")
        return 0.0

def detect_event_boundaries(video_path: str) -> List[int]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Video open failed: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        if duration < 10:
            params = {"n_clusters":1, "frame_diff_threshold":0.25, "motion_threshold":250}
        elif duration < 20:
            params = {"n_clusters":2, "frame_diff_threshold":0.25, "motion_threshold":250}
        elif duration < 30:
            params = {"n_clusters":3, "frame_diff_threshold":0.18, "motion_threshold":180}
        elif duration < 45:
            params = {"n_clusters":4, "frame_diff_threshold":0.12, "motion_threshold":120}
        else:
            params = {"n_clusters":5, "frame_diff_threshold":0.10, "motion_threshold":100}
        
        boundaries = []
        features = []
        prev_frame = None

        while True:
            ret, current_frame = cap.read()
            if not ret:
                break

            feature = extract_features(current_frame)
            if feature is not None:
                features.append(feature)

            if prev_frame is not None:
                frame_diff = compute_frame_difference(prev_frame, current_frame)
                motion_score = compute_motion_histogram(prev_frame, current_frame)
                if frame_diff > params["frame_diff_threshold"] and motion_score > params["motion_threshold"]:
                    boundaries.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

            prev_frame = current_frame

        if len(features) < params["n_clusters"]:
            return []

        features = np.array(features)
        kmeans = KMeans(n_clusters=params["n_clusters"], random_state=0, n_init=10).fit(features)
        labels = kmeans.labels_
        
        refined_boundaries = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
        refined_boundaries = [b for b in refined_boundaries if b - (refined_boundaries[refined_boundaries.index(b) - 1] if refined_boundaries.index(b) > 0 else 0) >= 80]
        
        final_boundaries = []
        for boundary in refined_boundaries:
            if not final_boundaries or (boundary - final_boundaries[-1] >= 80):
                final_boundaries.append(boundary)

        if len(final_boundaries) > 5:
            step = max(1, len(final_boundaries) // 5)
            final_boundaries = final_boundaries[::step][:5]
        elif not final_boundaries and refined_boundaries:
            final_boundaries = [refined_boundaries[len(refined_boundaries) // 2]]

        cap.release()
        return final_boundaries

    except Exception as e:
        print(f"Event detection failed: {str(e)}")
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
    
    parser = argparse.ArgumentParser(description="Video event segmentation tool")
    parser.add_argument("-i", "--input", required=True, help="Input JSON path")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    parser.add_argument("-d", "--data_root", default="./", help="Data root directory")
    
    args = parser.parse_args()
    
    build_dataset(
        input_json=args.input,
        output_json=args.output,
        base_dir=args.data_root
    )