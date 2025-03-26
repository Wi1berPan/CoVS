import json

with open('', 'r', encoding='utf-8') as f:
    new_data = json.load(f)

with open('', 'r', encoding='utf-8') as f:
    old_data = json.load(f)

old_dict = {item['id']: item for item in old_data}

for new_item in new_data:
    new_id = new_item['id']
    if new_id in old_dict:
        old_item = old_dict[new_id]
        for old_conv in old_item['conversations']:
            merged_conv = old_conv.copy()
            if merged_conv['from'] == 'human':
                merged_conv['value'] = merged_conv['value'].replace('<image>\n', '', 1)
            new_item['conversations'].append(merged_conv)

with open('', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

