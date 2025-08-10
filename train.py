import torch
from model.embeddings import EmbeddingNet
from model.meta_learner import MetaLearner
from model.utils import load_interaction_data, map_ids, create_few_shot_tasks
import os
import matplotlib.pyplot as plt

# 1. Load and prepare dataset
data_path = os.path.join("data", "interactions.csv")
data = load_interaction_data(data_path)
mapped_data, user2idx, item2idx = map_ids(data)

# 2. Create few-shot tasks
tasks = create_few_shot_tasks(mapped_data, k_shot=1)

# 3. Model setup
NUM_USERS = len(user2idx)
NUM_ITEMS = len(item2idx)
EMBED_DIM = 32

model = EmbeddingNet(NUM_USERS, NUM_ITEMS, EMBED_DIM)
model.args = (NUM_USERS, NUM_ITEMS, EMBED_DIM)  # Required for cloning in meta-learning
meta_learner = MetaLearner(model)

# 4. Prepare training tasks
training_tasks = []
for task in tasks:
    uid = task['user_id']
    support_items = [i[0] for i in task['support']]
    support_ratings = [i[1] for i in task['support']]
    query_items = [i[0] for i in task['query']]
    query_ratings = [i[1] for i in task['query']]

    if not query_items or not support_items:
        continue  # skip invalid tasks

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

    training_tasks.append((support_x, support_y, query_x, query_y))

print(f"✅ Total training tasks: {len(training_tasks)}")

# 5. Training loop
EPOCHS = 5
loss_history = []

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for support_x, support_y, query_x, query_y in training_tasks:
        loss = meta_learner.train_task(support_x, support_y, query_x, query_y)
        total_loss += loss
    avg_loss = total_loss / len(training_tasks)
    loss_history.append(avg_loss)
    print(f"📘 Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

# 6. Save model
torch.save(model.state_dict(), 'trained_model.pth')
print("✅ Trained model saved to trained_model.pth")

# 7. Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', linestyle='-')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Avg Loss")
plt.grid(True)
plt.savefig("loss_plot.png")
print("📈 Loss plot saved as loss_plot.png")
plt.show()
