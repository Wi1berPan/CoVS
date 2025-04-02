import json
import os
from collections import defaultdict
import cv2
from tqdm import tqdm  # 导入进度条库

def get_video_duration(video_path):
    """获取视频时长的实际实现（使用OpenCV）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_keyframes(a_json, b_json):
    # 确保两个JSON文件的条目数量相同
    if len(a_json) != len(b_json):
        raise ValueError("A_json和B_json的条目数量不匹配")
    
    # 定义时间区间
    time_ranges = [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 25),
        (25, 30)
    ]
    
    # 创建统计字典
    stats = {
        f"{low}-{high}": {
            'total_videos': 0,
            'total_keyframes': 0,
            'empty_keyframes': 0,
            'video_ids': []
        }
        for low, high in time_ranges
    }
    
    # 添加进度条
    progress_bar = tqdm(zip(a_json, b_json), total=len(a_json), desc="处理进度")
    
    # 按照顺序处理对应的条目
    for idx, (a_item, b_item) in enumerate(progress_bar):
        try:
            video_path = os.path.join('/data/P_Yih/huggingface/LLaVA-Video-178K', 
                                b_item['data_source'], 
                                b_item['video'].lstrip('/'))
            
            # 更新进度条描述
            progress_bar.set_description(f"处理 {idx+1}/{len(a_json)}: {video_path[-20:]}")
            
            # 获取视频时长
            duration = get_video_duration(video_path)
        except Exception as e:
            progress_bar.write(f"处理第{idx}条记录时出错: {str(e)}")
            continue
        
        # 确定时间区间
        time_range = None
        for low, high in time_ranges:
            if low <= duration < high:
                time_range = f"{low}-{high}"
                break
        
        if not time_range:
            continue
            
        # 更新统计信息
        stats[time_range]['total_videos'] += 1
        stats[time_range]['video_ids'].append(f"Pos_{idx}")
        
        keyframes = a_item['Key frames for event segmentation']
        if not keyframes:  # 空关键帧
            stats[time_range]['empty_keyframes'] += 1
        else:
            stats[time_range]['total_keyframes'] += len(keyframes)
    
    # 关闭进度条
    progress_bar.close()
    
    # 计算结果
    results = []
    for time_range, data in stats.items():
        if data['total_videos'] == 0:
            continue
            
        # 计算平均关键帧数（不包括空关键帧的视频）
        valid_videos = data['total_videos'] - data['empty_keyframes']
        avg_keyframes = data['total_keyframes'] / valid_videos if valid_videos > 0 else 0
        
        # 计算错误率（空关键帧的比例）
        error_rate = data['empty_keyframes'] / data['total_videos']
        
        results.append({
            'time_range': time_range,
            'total_videos': data['total_videos'],
            'average_keyframes': round(avg_keyframes, 2),
            'error_rate': round(error_rate, 4),
            'empty_keyframes_count': data['empty_keyframes'],
            'sample_positions': data['video_ids'][:3]
        })
    
    return results

if __name__ == "__main__":
    # 加载JSON文件
    a_json = load_json('/data/P_Yih/CoVS/data/Ablation_out_pre/HSV.json')
    b_json = load_json('/data/P_Yih/huggingface/LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed_wash.json')
    
    try:
        # 分析关键帧
        print("开始处理视频关键帧分析...")
        analysis_results = analyze_keyframes(a_json, b_json)
        
        # 打印结果
        print("\n视频关键帧分析结果:")
        print("{:<10} {:<15} {:<20} {:<15} {:<25} {:<20}".format(
            "时长区间", "视频总数", "平均关键帧数", "错误率", "空关键帧数量", "示例位置"))
        
        for result in analysis_results:
            print("{:<10} {:<15} {:<20} {:<15} {:<25} {:<20}".format(
                result['time_range'],
                result['total_videos'],
                result['average_keyframes'],
                f"{result['error_rate']*100:.2f}%",
                result['empty_keyframes_count'],
                str(result['sample_positions'])
            ))

        # 保存结果到JSON文件
        with open('keyframe_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print("\n分析结果已保存到 keyframe_analysis_results_canny.json")
            
    except ValueError as e:
        print(f"错误: {str(e)}")
    except Exception as e:
        print(f"发生意外错误: {str(e)}")