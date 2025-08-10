import torch
from sklearn.metrics import ndcg_score
from model.embeddings import EmbeddingNet
from model.utils import load_interaction_data, map_ids, create_few_shot_tasks
import os

# 📌 Hit Rate@k
def hit_rate(y_true, y_pred, k=5):
    top_k = torch.topk(y_pred, k).indices
    hits = (y_true[top_k] > 0).float()
    return hits.mean().item()

# ⚙️ Settings
DATA_PATH = 'data/interactions.csv'
MODEL_PATH = 'trained_model.pth'
K = 5  # Top-k

# 🔹 Load Data
data = load_interaction_data(DATA_PATH)
mapped_data, user2idx, item2idx = map_ids(data)
tasks = create_few_shot_tasks(mapped_data, k_shot=1)

NUM_USERS = len(user2idx)
NUM_ITEMS = len(item2idx)
EMBED_DIM = 32

# 🔹 Load Model
model = EmbeddingNet(NUM_USERS, NUM_ITEMS, EMBED_DIM)
model.args = (NUM_USERS, NUM_ITEMS, EMBED_DIM)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("✅ Loaded trained model from:", MODEL_PATH)
else:
    print("⚠️ No saved model found. Using randomly initialized model.")

model.eval()

# 🔹 Evaluation Loop
total_ndcg = 0
total_hr = 0
count = 0

for task in tasks:
    uid = task['user_id']
    support_items = [i[0] for i in task['support']]
    support_ratings = [i[1] for i in task['support']]
    query_items = [i[0] for i in task['query']]
    query_ratings = [i[1] for i in task['query']]

    if len(query_items) < K:
        continue

    # Create tensors
    support_user_tensor = torch.tensor([uid] * len(support_items))
    query_user_tensor = torch.tensor([uid] * len(query_items))

    support_x = torch.cat([
        model.user_embed(support_user_tensor),
        model.item_embed(torch.tensor(support_items))
    ], dim=1)

    query_x = torch.cat([
        model.user_embed(query_user_tensor),
        model.item_embed(torch.tensor(query_items))
    ], dim=1)

    support_y = torch.tensor(support_ratings, dtype=torch.float32)
    query_y = torch.tensor(query_ratings, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        preds = model.fc(query_x).view(-1)

        # Evaluate
        ndcg = ndcg_score([query_y.numpy()], [preds.numpy()])
        hr = hit_rate(query_y, preds, k=K)

        total_ndcg += ndcg
        total_hr += hr
        count += 1

# 🔹 Final Results
if count > 0:
    avg_ndcg = total_ndcg / count
    avg_hr = total_hr / count
    print(f"\n📊 Evaluation Results (based on {count} users):")
    print(f"🔹 NDCG@{K}: {avg_ndcg:.4f}")
    print(f"🔹 HR@{K}:   {avg_hr:.4f}")
else:
    print("⚠️ Not enough query data to evaluate.")
