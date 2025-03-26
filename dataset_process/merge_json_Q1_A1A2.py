import json

# 读取两个JSON文件
with open('./0_30_s_academic_v0_1_sub_event_gpt.json', 'r', encoding='utf-8') as f:
    new_data = json.load(f)

with open('/data/P_Yih/huggingface/LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_v0_1_cap_processed_wash.json', 'r', encoding='utf-8') as f:
    old_data = json.load(f)

# 将Old数据按id映射到字典
old_dict = {item['id']: item for item in old_data}

# 遍历New数据并替换
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
                    # 如果是human，完全替换value并去掉<image>\n
                    if new_conv['from'] == 'human':
                        new_conv['value'] = old_conv['value']
                    # 如果是gpt，保持不变
                    # 如果需要替换gpt的value，可以取消以下注释
                    # else:
                    #     new_conv['value'] = old_conv['value']

# 保存合并后的结果到新文件（避免覆盖原文件）
with open('0_30_s_academic_v0_1_cap_processed_Q2_A1A2.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print("合并完成，结果已保存")