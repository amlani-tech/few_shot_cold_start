import csv
import random
from collections import defaultdict
from sklearn.cluster import KMeans
import torch

def load_interaction_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            rating = float(row['rating'])
            data.append((user_id, item_id, rating))
    return data

def map_ids(data):
    user_ids = sorted(set([d[0] for d in data]))
    item_ids = sorted(set([d[1] for d in data]))
    user2idx = {uid: i for i, uid in enumerate(user_ids)}
    item2idx = {iid: i for i, iid in enumerate(item_ids)}
    mapped_data = [(user2idx[u], item2idx[i], r) for u, i, r in data]
    return mapped_data, user2idx, item2idx

def create_few_shot_tasks(data, k_shot=1):
    user_data = defaultdict(list)
    for user_id, item_id, rating in data:
        user_data[user_id].append((item_id, rating))

    tasks = []
    for user_id, interactions in user_data.items():
        if len(interactions) <= k_shot:
            continue
        random.shuffle(interactions)
        support = interactions[:k_shot]
        query = interactions[k_shot:]
        tasks.append({
            'user_id': user_id,
            'support': support,
            'query': query
        })
    return tasks

def cluster_users(model, user_ids, num_clusters=3):
    with torch.no_grad():
        user_tensor = torch.tensor(user_ids)
        user_embeds = model.user_embed(user_tensor).numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(user_embeds)
    return {user: cluster for user, cluster in zip(user_ids, cluster_labels)}
from sklearn.cluster import KMeans

def cluster_users(model, user_ids, num_clusters=5):
    with torch.no_grad():
        embeddings = model.user_embed(torch.tensor(user_ids)).numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
    return kmeans.labels_
