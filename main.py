# main.py

from model.utils import load_interaction_data, map_ids, create_few_shot_tasks

data = load_interaction_data('data/interactions.csv')
data, user2idx, item2idx = map_ids(data)
tasks = create_few_shot_tasks(data, k_shot=1)

print(f"Loaded {len(tasks)} tasks.")
for task in tasks[:2]:
    print(task)
