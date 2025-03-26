import json
import os
import argparse

def format_json_file(input_file_path):
    try:
        base_name, ext = os.path.splitext(input_file_path)
        output_file_path = f"{base_name}_wash{ext}"

        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        formatted_json = json.dumps(data, indent=4, ensure_ascii=False)

        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(formatted_json)

        print(f"Successfully formatted and saved: {input_file_path} -> {output_file_path}")
    except FileNotFoundError:
        print(f"Error: File not found {input_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON file {input_file_path}")
    except Exception as e:
        print(f"Unknown error occurred while processing {input_file_path}: {e}")

def process_all_json_files(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                format_json_file(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format all JSON files in specified directory')
    parser.add_argument('directory', nargs='?', default='.',
                        help='Directory path to process (default: current directory)')
    args = parser.parse_args()

    target_directory = args.directory
    process_all_json_files(target_directory)