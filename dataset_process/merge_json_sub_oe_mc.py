import json

# 读取两个JSON文件v
with open('./0_30_s_academic_v0_1_sub_event_gpt.json', 'r', encoding='utf-8') as f:
    new_data = json.load(f)

with open('/data/P_Yih/huggingface/LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_mc_v0_1_qa_processed.json', 'r', encoding='utf-8') as f:
    old_data = json.load(f)

# 将Old数据按id分组（因为Old中可能有重复id）
from collections import defaultdict

old_dict = defaultdict(list)
for item in old_data:
    old_dict[item['id']].append(item)

# 遍历New数据并合并
merged_data = []
for new_item in new_data:
    new_id = new_item['id']
    if new_id in old_dict:
        # 找到Old中所有对应id的单元
        old_items = old_dict[new_id]
        # 将Old的conversations拼接到New的conversations中
        for old_item in old_items:
            new_item['conversations'].extend(old_item['conversations'])
        # 将合并后的单元添加到结果中
        merged_data.append(new_item)

# 保存合并后的结果到新文件
with open('0_30_s_academic_mc_v0_1_qa_processed_wash_Q2_mcqa.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print("合并完成，结果已保存到 Merged.json")

