#usage
#python build_json_new.py -i ../0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed_wash.json -o ../0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed_wash_pca_dynamic_0.3.json -d /data/P_Yih/huggingface/LLaVA-Video-178K
import os
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List, Dict
#import * from GEBD

def extract_features(frame: np.ndarray) -> np.ndarray:
    """
    提取帧的特征（颜色直方图 + 边缘信息）
    :param frame: 输入帧（BGR格式）
    :return: 合并后的特征向量
    """
    try:
        # 颜色直方图（HSV空间）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()

        # 边缘信息（Canny边缘检测）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()

        # 合并特征
        features = np.hstack([color_hist, edge_hist])
        return features
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return None

def compute_frame_difference(prev: np.ndarray, curr: np.ndarray) -> float:
    """
    计算两帧之间的差异（标准化到0-1）
    :param prev: 前一帧（BGR格式）
    :param curr: 当前帧（BGR格式）
    :return: 帧间差异值（0-1）
    """
    try:
        # 转换为灰度图
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # 计算绝对差异
        diff = cv2.absdiff(prev_gray, curr_gray)

        # 二值化处理
        _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # 计算差异比例（标准化到0-1）
        diff_ratio = np.sum(diff) / (diff.size * 255)
        return diff_ratio
    except Exception as e:
        print(f"帧差计算失败: {str(e)}")
        return 0.0
    
def compute_motion_histogram(prev: np.ndarray, curr: np.ndarray) -> float:
    """
    计算运动直方图（基于稀疏光流）
    :param prev: 前一帧（BGR格式）
    :param curr: 当前帧（BGR格式）
    :return: 运动量（标准化值）
    """
    try:
        # 转换为灰度图
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # 检测角点
        feature_params = dict(
            maxCorners=200,       # 最大角点数
            qualityLevel=0.3,      # 角点质量阈值
            minDistance=7,         # 角点间最小距离
            blockSize=7           # 角点检测窗口大小
        )
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if p0 is None:
            return 0.0

        # 计算光流
        p1, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)

        # 计算运动向量
        motion_vectors = np.linalg.norm(p1 - p0, axis=2)

        # 统计运动量
        motion_hist = np.histogram(motion_vectors, bins=10, range=(0, 10))[0]
        motion_score = np.sum(motion_hist) / (len(p0) * 10)  # 标准化到0-1
        return motion_score
    except Exception as e:
        print(f"运动检测失败: {str(e)}")
        return 0.0
    
def detect_event_boundaries(video_path: str) -> List[int]:
    """
    检测视频中的事件边界
    :param video_path: 视频文件路径
    :return: 关键帧列表
    """
    try:
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频参数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # 根据时长动态调整参数
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
        print(f"视频时长: {duration}, 参数: {params}")
        
        # 初始化变量
        boundaries = []
        features = []
        frames = []
        prev_frame = None

        # 逐帧处理
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break

            # 提取特征
            feature = extract_features(current_frame)
            if feature is not None:
                features.append(feature)
                frames.append(current_frame)

            # 计算帧差和运动量
            if prev_frame is not None:
                frame_diff = compute_frame_difference(prev_frame, current_frame)
                motion_score = compute_motion_histogram(prev_frame, current_frame)

                # 检测事件边界
                if frame_diff > params["frame_diff_threshold"] and motion_score > params["motion_threshold"]:
                    boundaries.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

            prev_frame = current_frame

        # 聚类分析
        if len(features) < params["n_clusters"]:
            print("有效帧数不足，跳过聚类分析")
            return []

        features = np.array(features)
        kmeans = KMeans(n_clusters=params["n_clusters"], random_state=0, n_init=10).fit(features)
        labels = kmeans.labels_
        
        refined_boundaries = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
        min_duration = 80  # 最小事件持续时间（帧数）
        refined_boundaries = [b for b in refined_boundaries if b - (refined_boundaries[refined_boundaries.index(b) - 1] if refined_boundaries.index(b) > 0 else 0) >= min_duration]
        
        

        # 根据聚类结果调整边界
        refined_boundaries = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                refined_boundaries.append(i)

        # 过滤间隔过小的边界
        final_boundaries = []
        for boundary in refined_boundaries:
            if not final_boundaries or (boundary - final_boundaries[-1] >= 80):
                final_boundaries.append(boundary)

        # 确保边界数量在1-5之间
        if len(final_boundaries) > 5:
            step = max(1, len(final_boundaries) // 5)
            final_boundaries = final_boundaries[::step][:5]
        elif not final_boundaries and refined_boundaries:
            final_boundaries = [refined_boundaries[len(refined_boundaries) // 2]]

        cap.release()
        return final_boundaries

    except Exception as e:
        print(f"事件边界检测失败: {str(e)}")
        return []


def process_entry(entry: Dict, base_dir: str = "./") -> Dict:
    """
    处理单个JSON条目
    :param entry: 原始数据条目
    :param base_dir: 数据根目录
    :return: 处理后的新格式条目
    """
    # 构建完整视频路径
    video_rel_path = os.path.join(
        entry["data_source"], 
        entry["video"]
    )
    video_full_path = os.path.abspath(os.path.join(base_dir, video_rel_path))
    print(f"视频文件路径: {video_full_path}")
    
    # 验证文件存在性
    if not os.path.exists(video_full_path):
        print(f"视频文件缺失: {video_full_path}")
        return None

    try:
        # 执行事件分割
        boundaries = detect_event_boundaries(video_full_path)
        
        # 构建结果条目
        video_id = os.path.splitext(os.path.basename(entry["video"]))[0]
        
        return {
            "id": video_id,
            "number_sub_event": len(boundaries) + 1,  # 子事件数 = 关键帧数 + 1
            "Key frames for event segmentation": boundaries
        }
    except Exception as e:
        print(f"处理视频 {video_full_path} 失败: {str(e)}")
        return None

def build_dataset(input_json: str, output_json: str, base_dir: str = "./"):
    """
    主处理流程
    :param input_json: 输入JSON路径
    :param output_json: 输出JSON路径
    :param base_dir: 数据根目录（默认为当前目录）
    """
    # 检查 JSON 文件是否存在
    if not os.path.exists(input_json):
        print(f"JSON 文件不存在: {input_json}")
        return

    # 读取原始数据
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"成功加载 JSON 文件: {input_json}")
    except Exception as e:
        print(f"加载 JSON 文件失败: {str(e)}")
        return

    # 处理所有条目
    results = []
    for idx, entry in enumerate(dataset):
        print(f"处理进度: {idx+1}/{len(dataset)}")
        processed = process_entry(entry, base_dir)
        if processed:
            results.append(processed)
    
    # 保存结果
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"处理完成！有效处理 {len(results)}/{len(dataset)} 个视频")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="视频事件分割数据集构建工具")
    parser.add_argument("-i", "--input", required=True, help="输入JSON路径")
    parser.add_argument("-o", "--output", required=True, help="输出JSON路径")
    parser.add_argument("-d", "--data_root", default="./", help="数据根目录路径")
    
    args = parser.parse_args()
    
    print(f"当前工作目录: {os.getcwd()}")
    print(f"输入文件路径: {os.path.abspath(args.input)}")
    
    build_dataset(
        input_json=args.input,
        output_json=args.output,
        base_dir=args.data_root
    )


    # 使用方法：
    #python build_json.py -i ../0_30_s_academic_v0_1/test.json -o ../0_30_s_academic_v0_1/output_test.json -d /data/P_Yih/huggingface/LLaVA-Video-178K