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
            for new_conv in new_item['conversations']:
                if new_conv['from'] == old_conv['from']:
                    if new_conv['from'] == 'human':
                        old_value = old_conv['value'].replace('<image>\n', '', 1)
                    else:
                        old_value = old_conv['value']
                    new_conv['value'] += '\n' + old_value

with open('', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)
