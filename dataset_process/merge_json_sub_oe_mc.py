import json
from collections import defaultdict

with open('', 'r', encoding='utf-8') as f:
    new_data = json.load(f)

with open('', 'r', encoding='utf-8') as f:
    old_data = json.load(f)

old_dict = defaultdict(list)
for item in old_data:
    old_dict[item['id']].append(item)

merged_data = []
for new_item in new_data:
    new_id = new_item['id']
    if new_id in old_dict:
        old_items = old_dict[new_id]
        for old_item in old_items:
            new_item['conversations'].extend(old_item['conversations'])
        merged_data.append(new_item)

with open('', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

