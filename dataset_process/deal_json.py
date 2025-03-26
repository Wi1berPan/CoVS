#增加json可读性

import json
import os
import argparse

def format_json_file(input_file_path):
    """
    格式化单个 JSON 文件
    :param input_file_path: 输入的 JSON 文件路径
    """
    try:
        # 构建输出文件路径
        base_name, ext = os.path.splitext(input_file_path)
        output_file_path = f"{base_name}_wash{ext}"

        # 打开并读取 JSON 文件
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 整理 JSON 数据，设置缩进为 4 个空格
        formatted_json = json.dumps(data, indent=4, ensure_ascii=False)

        # 将整理后的 JSON 数据写入新文件
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(formatted_json)

        print(f"成功整理并保存文件: {input_file_path} -> {output_file_path}")
    except FileNotFoundError:
        print(f"错误: 未找到文件 {input_file_path}")
    except json.JSONDecodeError:
        print(f"错误: 无法解析 JSON 文件 {input_file_path}")
    except Exception as e:
        print(f"处理文件 {input_file_path} 时发生未知错误: {e}")

def process_all_json_files(directory):
    """
    处理指定目录下的所有 JSON 文件
    :param directory: 要处理的目录路径
    """
    if not os.path.exists(directory):
        print(f"错误: 目录 {directory} 不存在")
        return

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                format_json_file(file_path)

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='整理指定文件夹下的所有 JSON 文件')
    parser.add_argument('directory', nargs='?', default='.',
                        help='要处理的文件夹路径，默认为当前目录')
    args = parser.parse_args()

    target_directory = args.directory
    process_all_json_files(target_directory)