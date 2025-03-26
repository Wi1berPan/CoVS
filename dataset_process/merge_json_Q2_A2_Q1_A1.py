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
        # 处理Old的对话内容
        for old_conv in old_item['conversations']:
            # 复制对话结构
            merged_conv = old_conv.copy()
            # 处理Human消息的value
            if merged_conv['from'] == 'human':
                # 移除开头的<image>\n
                merged_conv['value'] = merged_conv['value'].replace('<image>\n', '', 1)
            # 将处理后的对话添加到New的conversations中
            new_item['conversations'].append(merged_conv)

# 保存合并后的结果到新文件（避免覆盖原文件）
with open('0_30_s_academic_v0_1_cap_processed_wash_Q2_A2_Q1_A1.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print("合并完成，结果已保存到 Merged.json")