import streamlit as st
import torch
import pandas as pd
import os
from sklearn.metrics import ndcg_score
from model.embeddings import EmbeddingNet
from model.utils import load_interaction_data, map_ids, create_few_shot_tasks

# ========== ⚙️ Utility Functions ==========
@st.cache_data
def load_data(path):
    data = load_interaction_data(path)
    return map_ids(data)

def hit_rate(y_true, y_pred, k=5):
    top_k = torch.topk(y_pred, k).indices
    hits = (y_true[top_k] > 0).float()
    return hits.mean().item()

# ========== 🚀 Load Model ==========
st.title("🎯  Few-Shot Learning for Cold-Start News Recommendation")

MODEL_PATH = "trained_model.pth"
DATA_PATH = "data/interactions.csv"

# 📁 Upload CSV
uploaded_file = st.file_uploader("Upload your own interactions.csv (optional)", type=["csv"])
if uploaded_file:
    with open("data/custom.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    DATA_PATH = "data/custom.csv"
    st.success("✅ Custom data loaded!")

# 🔄 Load mapped data
mapped_data, user2idx, item2idx = load_data(DATA_PATH)
NUM_USERS = len(user2idx)
NUM_ITEMS = len(item2idx)
EMBED_DIM = 32

# 🧠 Model
model = EmbeddingNet(NUM_USERS, NUM_ITEMS, EMBED_DIM)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ========== 🎯 Single Prediction ==========
st.subheader("🔍 Predict Single Rating")

user_id = st.number_input("User ID", min_value=0, max_value=NUM_USERS - 1, value=0)
item_id = st.number_input("Item ID", min_value=0, max_value=NUM_ITEMS - 1, value=0)

if st.button("Predict Rating"):
    with torch.no_grad():
        user_tensor = torch.tensor([user_id])
        item_tensor = torch.tensor([item_id])
        x = torch.cat([
            model.user_embed(user_tensor),
            model.item_embed(item_tensor)
        ], dim=1)
        pred = model.fc(x).item()
    st.success(f"⭐ Predicted Rating: {round(pred, 2)} / 5")

# ========== 📊 Evaluate Full Dataset ==========
st.subheader("📈 Evaluate Model (NDCG & HR@k)")

if st.button("Run Evaluation"):
    tasks = create_few_shot_tasks(mapped_data, k_shot=1)

    total_ndcg = 0
    total_hr = 0
    count = 0
    k = 5
    predictions = []

    for task in tasks:
        uid = task['user_id']
        support = task['support']
        query = task['query']

        if len(query) < k:
            continue

        support_items = [x[0] for x in support]
        support_ratings = [x[1] for x in support]
        query_items = [x[0] for x in query]
        query_ratings = [x[1] for x in query]

        support_x = torch.cat([
            model.user_embed(torch.tensor([uid] * len(support_items))),
            model.item_embed(torch.tensor(support_items))
        ], dim=1)
        query_x = torch.cat([
            model.user_embed(torch.tensor([uid] * len(query_items))),
            model.item_embed(torch.tensor(query_items))
        ], dim=1)

        support_y = torch.tensor(support_ratings, dtype=torch.float32)
        query_y = torch.tensor(query_ratings, dtype=torch.float32)

        with torch.no_grad():
            preds = model.fc(query_x).view(-1)

        ndcg = ndcg_score([query_y.numpy()], [preds.numpy()])
        hr = hit_rate(query_y, preds, k)

        total_ndcg += ndcg
        total_hr += hr
        count += 1

        predictions.append({
            "user_id": uid,
            "query_items": query_items,
            "true_ratings": query_ratings,
            "predicted_ratings": preds.numpy().tolist()
        })

    if count > 0:
        avg_ndcg = total_ndcg / count
        avg_hr = total_hr / count
        st.success(f"✅ Evaluation Complete!")
        st.markdown(f"- **Average NDCG@{k}**: `{avg_ndcg:.4f}`")
        st.markdown(f"- **Average HR@{k}**: `{avg_hr:.4f}`")

        # Save results
        df_pred = pd.DataFrame(predictions)
        df_pred.to_json("results.json", orient="records", indent=2)
        st.download_button("📥 Download Predictions JSON", data=open("results.json", "rb"), file_name="results.json")
    else:
        st.warning("⚠️ Not enough query data to evaluate.")
