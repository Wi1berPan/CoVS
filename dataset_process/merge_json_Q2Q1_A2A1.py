import json

# 读取两个JSON文件
with open('./0_30_s_academic_v0_1_sub_event_gpt.json', 'r', encoding='utf-8') as f:
    new_data = json.load(f)

with open('/data/P_Yih/huggingface/LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed_wash.json', 'r', encoding='utf-8') as f:
    old_data = json.load(f)

# 将Old数据按id映射到字典
old_dict = {item['id']: item for item in old_data}

# 遍历New数据并合并
for new_item in new_data:
    new_id = new_item['id']
    if new_id in old_dict:
        old_item = old_dict[new_id]
        # 遍历Old的对话内容
        for old_conv in old_item['conversations']:
            # 找到New中对应的对话
            for new_conv in new_item['conversations']:
                # 匹配角色（human或gpt）
                if new_conv['from'] == old_conv['from']:
                    # 如果是human，去掉<image>\n后追加
                    if new_conv['from'] == 'human':
                        old_value = old_conv['value'].replace('<image>\n', '', 1)
                    else:
                        old_value = old_conv['value']
                    # 将Old的value追加到New的value中
                    new_conv['value'] += '\n' + old_value

# 保存合并后的结果到新文件（避免覆盖原文件）
with open('0_30_s_academic_v0_1_cap_processed_Q2Q1_A2A1.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print("合并完成，结果已保存到 0_30_s_academic_v0_1_Merge_json_Q1Q2_A1A2.json")