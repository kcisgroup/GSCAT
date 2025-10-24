import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- 1. CONFIGURATION ---
class Config:
    BASE_DIR = os.getcwd()
    DATA_PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_tky_cleaned.csv')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    EMBED_SIZE = 128
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    EPOCHS = 30
    WEIGHT_DECAY = 1e-5
    MIN_TRAJ_LEN = 3

    TOP_K = [1, 5, 10]

# --- 2. MOBILITY TREE HELPER CLASS (Paste the MobilityTree class from above here) ---
class MobilityTree:
    def __init__(self):
        self.num_nodes = 33
        self.time_to_leaf_map = {}
        for h in range(24):
            self.time_to_leaf_map[(0, h)] = 9 + h
            self.time_to_leaf_map[(1, h)] = 9 + h
        self.child_to_parent = {}
        for h in range(6): self.child_to_parent[9 + h] = 5
        for h in range(6, 11): self.child_to_parent[9 + h] = 3
        for h in range(11, 14): self.child_to_parent[9 + h] = 4
        for h in range(14, 18): self.child_to_parent[9 + h] = 6
        for h in range(18, 22): self.child_to_parent[9 + h] = 7
        for h in range(22, 24): self.child_to_parent[9 + h] = 8
        self.child_to_parent[1] = 0
        self.child_to_parent[2] = 0

    def get_path_to_root(self, hour, is_weekend):
        leaf_node = self.time_to_leaf_map.get((is_weekend, hour), self.time_to_leaf_map[(0,0)])
        path = [leaf_node]
        timeofday_node = self.child_to_parent.get(leaf_node)
        if timeofday_node is not None:
            path.append(timeofday_node)
            daytype_node = 2 if is_weekend else 1
            path.append(daytype_node)
            path.append(0) # Root
        return path

# --- 3. DATA PREPROCESSING ---
def preprocess_data_for_mtnet(config, tree):
    print("Step 1: Loading and mapping data...")
    df = pd.read_csv(config.DATA_PATH)
    df.rename(columns={'geo_id': 'poi_id'}, inplace=True)
    
    # Create mappings
    user_map = {user: i for i, user in enumerate(df['user_id'].unique())}
    poi_map = {poi: i for i, poi in enumerate(df['poi_id'].unique())}
    n_users = len(user_map)
    n_pois = len(poi_map)

    df['user_id'] = df['user_id'].map(user_map)
    df['poi_id'] = df['poi_id'].map(poi_map)
    
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['user_id', 'time'], inplace=True)

    print("Step 2: Splitting user trajectories and generating samples (8:1:1)...")
    train_data, valid_data, test_data = [], [], []
    user_groups = df.groupby('user_id')

    for _, group in tqdm(user_groups, desc="Processing users"):
        pois = group['poi_id'].tolist()
        hours = group['hour'].tolist()
        is_weekends = group['is_weekend'].tolist()
        
        num_checkins = len(pois)

        # Ensure user has enough check-ins for a meaningful split
        if num_checkins < 5: # Threshold can be adjusted
            continue # Skip users with very short histories

        # Calculate split indices based on the length of the trajectory
        train_end_idx = int(num_checkins * 0.8)
        valid_end_idx = int(num_checkins * 0.9)
        
        # Generate samples for each set
        # The split is based on the target item's position in the sequence
        for i in range(num_checkins - 1): # Iterate to create (last, target) pairs
            sample = {
                'user': group.iloc[i]['user_id'],
                'last_poi': pois[i],
                'target_poi': pois[i+1],
                'target_hour': hours[i+1],
                'target_is_weekend': is_weekends[i+1]
            }
            
            target_idx = i + 1
            if target_idx < train_end_idx:
                train_data.append(sample)
            elif target_idx < valid_end_idx:
                valid_data.append(sample)
            else:
                test_data.append(sample)
        
    return train_data, valid_data, test_data, n_users, n_pois

# --- 4. PYTORCH DATASET ---
class MTNetDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# --- 5. MTNet MODEL ---
class MTNet(nn.Module):
    def __init__(self, n_users, n_pois, tree, config):
        super(MTNet, self).__init__()
        self.n_users = n_users
        self.n_pois = n_pois
        self.tree = tree
        self.embed_size = config.EMBED_SIZE

        # Standard embeddings
        self.poi_embedding = nn.Embedding(n_pois, self.embed_size)
        
        # --- Core of MTNet: User preferences stored in the tree structure ---
        # Shape: (num_users, num_tree_nodes, embed_size)
        self.user_preference_tree = nn.Embedding(n_users * tree.num_nodes, self.embed_size)
        # We can reshape the user_id to access this: idx = user_id * num_nodes + node_id

        # Attention mechanism for top-down selection
        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size // 2),
            nn.ReLU(),
            nn.Linear(self.embed_size // 2, 1)
        )

    def forward(self, user, last_poi, target_hour, target_is_weekend):
        # user: (B), last_poi: (B), target_hour: (B), target_is_weekend: (B)
        batch_size = user.size(0)
        
        # 1. Get the embedding for the last visited POI (context)
        last_poi_embed = self.poi_embedding(last_poi) # (B, D)

        # 2. Top-Down Preference Selection
        # For each item in the batch, get the active path in the mobility tree
        paths = [self.tree.get_path_to_root(h, w) for h, w in zip(target_hour.tolist(), target_is_weekend.tolist())]
        path_lengths = [len(p) for p in paths]
        max_path_len = max(path_lengths)

        # Prepare tensors for batch-wise preference lookup and attention
        path_nodes_tensor = torch.zeros(batch_size, max_path_len, dtype=torch.long, device=user.device)
        path_mask = torch.zeros(batch_size, max_path_len, dtype=torch.bool, device=user.device)

        for i, p in enumerate(paths):
            path_nodes_tensor[i, :len(p)] = torch.tensor(p, device=user.device)
            path_mask[i, :len(p)] = True

        # Look up preference embeddings for all nodes in all paths
        # Shape user_ids for broadcasting: (B) -> (B, 1) -> (B, max_path_len)
        user_ids_expanded = user.unsqueeze(1).expand(-1, max_path_len)
        
        # Calculate flat indices for the embedding lookup
        flat_indices = user_ids_expanded * self.tree.num_nodes + path_nodes_tensor
        
        # Get preference vectors: (B, max_path_len, D)
        path_preference_embeds = self.user_preference_tree(flat_indices)

        # 3. Attention Calculation
        # The last POI embedding acts as the query for the attention mechanism
        query = last_poi_embed.unsqueeze(1) # (B, 1, D)
        
        # Simple additive attention
        attn_scores = self.attention_layer(torch.tanh(query + path_preference_embeds)).squeeze(-1) # (B, max_path_len)
        attn_scores.masked_fill_(~path_mask, -1e9) # Mask out padding
        
        attn_weights = F.softmax(attn_scores, dim=1) # (B, max_path_len)

        # 4. Compute the final time-aware user preference
        # Weighted sum of preference vectors on the path
        time_aware_user_pref = torch.bmm(attn_weights.unsqueeze(1), path_preference_embeds).squeeze(1) # (B, D)

        # 5. Prediction
        # The score is the dot product between the user preference and all POI embeddings
        all_poi_embeds = self.poi_embedding.weight # (n_pois, D)
        scores = torch.matmul(time_aware_user_pref, all_poi_embeds.t()) # (B, n_pois)

        return scores

# --- 6. TRAIN & EVALUATE ---
def train_mtnet(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_data in tqdm(dataloader, desc="Training"):
        user = batch_data['user'].to(device)
        last_poi = batch_data['last_poi'].to(device)
        target_poi = batch_data['target_poi'].to(device)
        target_hour = batch_data['target_hour'].to(device)
        target_is_weekend = batch_data['target_is_weekend'].to(device)

        optimizer.zero_grad()
        scores = model(user, last_poi, target_hour, target_is_weekend)
        loss = criterion(scores, target_poi)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, top_k, device):
    model.eval()
    metrics = {f'acc@{k}': 0 for k in top_k}
    metrics['mrr'] = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating"):
            user = batch_data['user'].to(device)
            last_poi = batch_data['last_poi'].to(device)
            target = batch_data['target_poi'].to(device)
            target_hour = batch_data['target_hour'].to(device)
            target_is_weekend = batch_data['target_is_weekend'].to(device)

            logits = model(user, last_poi, target_hour, target_is_weekend)
            max_k = max(top_k)
            _, top_indices = torch.topk(logits, max_k, dim=1)
            expanded_target = target.view(-1, 1).expand_as(top_indices)
            hits_matrix = (top_indices == expanded_target)
            for k in top_k:
                metrics[f'acc@{k}'] += hits_matrix[:, :k].any(dim=1).sum().item()
            match_indices = torch.nonzero(hits_matrix, as_tuple=True)
            reciprocal_ranks = torch.zeros(target.size(0)).to(device)
            ranks = match_indices[1].float() + 1.0
            reciprocal_ranks.scatter_(0, match_indices[0], 1.0 / ranks)
            metrics['mrr'] += reciprocal_ranks.sum().item()
            total_samples += target.size(0)
    for k in top_k:
        metrics[f'acc@{k}'] /= total_samples
    metrics['mrr'] /= total_samples
    return metrics

# --- 7. MAIN SCRIPT ---
if __name__ == '__main__':
    config = Config()
    tree = MobilityTree()
    print(f"Using device: {config.DEVICE}")

    # 1. PREPROCESS DATA
    # The function now returns three datasets
    train_data, valid_data, test_data, n_users, n_pois = preprocess_data_for_mtnet(config, tree)
    
    print(f"\nNum users: {n_users}, Num POIs: {n_pois}")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")

    # 2. CREATE DATALOADERS
    train_dataset = MTNetDataset(train_data)
    valid_dataset = MTNetDataset(valid_data)
    test_dataset = MTNetDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 3. INITIALIZE MODEL, OPTIMIZER, CRITERION
    model = MTNet(n_users, n_pois, tree, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # 4. TRAINING LOOP WITH VALIDATION & EARLY STOPPING
    best_val_mrr = 0.0
    best_model_state = None
    patience = 5  # Number of epochs to wait for improvement before stopping
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        
        # Train on the training set
        avg_loss = train_mtnet(model, train_loader, optimizer, criterion, config.DEVICE)
        print(f"Average Training Loss: {avg_loss:.4f}")
        
        # Evaluate on the validation set
        val_metrics = evaluate(model, valid_loader, config.TOP_K, config.DEVICE)
        print("--- Validation Metrics ---")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Check for improvement
        current_val_mrr = val_metrics['mrr']
        if current_val_mrr > best_val_mrr:
            best_val_mrr = current_val_mrr
            # Save the state of the best model
            best_model_state = model.state_dict()
            patience_counter = 0
            print("Validation MRR improved! Saving model state.")
        else:
            patience_counter += 1
            print(f"No improvement in validation MRR for {patience_counter} epoch(s).")

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # 5. FINAL EVALUATION ON TEST SET
    print("\n--- Training Finished ---")
    if best_model_state:
        print("Loading best model from validation phase for final testing...")
        model.load_state_dict(best_model_state)
    else:
        print("No best model was saved. Using the last model state for testing.")

    print("\n--- Final Test Metrics ---")
    test_metrics = evaluate(model, test_loader, config.TOP_K, config.DEVICE)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")