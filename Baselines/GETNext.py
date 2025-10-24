import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- 1. 配置参数 ---
class Config:
    BASE_DIR = os.getcwd()
    DATA_PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_tky_cleaned.csv')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模型参数
    EMBED_SIZE = 128
    N_HEADS = 4
    N_LAYERS = 2
    DROPOUT = 0.3
    
    # 训练参数
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 30
    WEIGHT_DECAY = 1e-5

    # 数据处理参数
    SESSION_LEN = 20
    MIN_TRAJ_LEN = 3

    # 评估参数
    TOP_K = [1, 5, 10]

# --- 2. 数据预处理 ---
def preprocess_data_for_getnext(config):
    print("Step 1: Loading and mapping data...")
    df = pd.read_csv(config.DATA_PATH)
    df.rename(columns={'geo_id': 'poi_id'}, inplace=True)
    
    poi_map = {poi: i + 1 for i, poi in enumerate(df['poi_id'].unique())}
    n_pois = len(poi_map) + 1
    
    df['poi_id_mapped'] = df['poi_id'].map(poi_map).fillna(0).astype(int)
    df.sort_values(by=['user_id', 'time'], inplace=True)

    # --- Step 2: Build Trajectory Flow Map ---
    print("Step 2: Building Trajectory Flow Map...")
    # 使用稀疏矩阵高效构建
    transition_counts = sp.dok_matrix((n_pois, n_pois), dtype=np.float32)
    user_groups = df.groupby('user_id')
    for _, group in tqdm(user_groups, desc="Building Flow Map"):
        traj = group['poi_id_mapped'].tolist()
        for i in range(len(traj) - 1):
            u, v = traj[i], traj[i+1]
            if u != 0 and v != 0: # 忽略 padding
                transition_counts[u, v] += 1
    
    # 转换为转移概率矩阵
    transition_counts = transition_counts.tocsr()
    row_sums = transition_counts.sum(axis=1)
    row_sums[row_sums == 0] = 1 # 防止除以零
    transition_probs = transition_counts.multiply(1.0 / row_sums)
    # 转换为稠密 PyTorch 张量
    flow_map_tensor = torch.from_numpy(transition_probs.toarray())

    # --- Step 3: Generate samples and split (8:1:1) ---
    print("Step 3: Generating samples and splitting...")
    train_data, valid_data, test_data = [], [], []
    for _, group in tqdm(user_groups, desc="Generating Samples"):
        traj = group['poi_id_mapped'].tolist()
        num_checkins = len(traj)
        if num_checkins < config.MIN_TRAJ_LEN: continue
            
        train_end_idx = int(num_checkins * 0.8)
        valid_end_idx = int(num_checkins * 0.9)
            
        for i in range(1, num_checkins):
            start_idx = max(0, i - config.SESSION_LEN)
            session = traj[start_idx:i]
            target = traj[i]
            if not session: continue

            sample = {'session': session, 'target': target}
            if i < train_end_idx: train_data.append(sample)
            elif i < valid_end_idx: valid_data.append(sample)
            else: test_data.append(sample)
        
    return train_data, valid_data, test_data, n_pois, flow_map_tensor

# --- 3. Dataset 和 DataLoader ---
class GETNextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        # 我们需要 last_poi 来查询 flow_map
        return {
            'session': torch.tensor(item['session'], dtype=torch.long),
            'last_poi': torch.tensor(item['session'][-1], dtype=torch.long),
            'target': torch.tensor(item['target'], dtype=torch.long)
        }

def collate_fn(batch):
    sessions = [item['session'] for item in batch]
    last_pois = torch.stack([item['last_poi'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    padded_sessions = pad_sequence(sessions, batch_first=True, padding_value=0)
    mask = (padded_sessions != 0)
    return {'session': padded_sessions, 'mask': mask, 'last_poi': last_pois, 'target': targets}

# --- 4. GETNext 模型定义 ---
class GETNext(nn.Module):
    def __init__(self, n_pois, flow_map_tensor, config):
        super(GETNext, self).__init__()
        self.config = config
        
        # 嵌入层
        self.poi_embedding = nn.Embedding(n_pois, config.EMBED_SIZE, padding_idx=0)
        self.pos_embedding = nn.Embedding(config.SESSION_LEN, config.EMBED_SIZE)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_SIZE, nhead=config.N_HEADS,
            dim_feedforward=config.EMBED_SIZE * 4, dropout=config.DROPOUT, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.N_LAYERS)
        
        # 局部预测层 (from Transformer)
        self.fc_local = nn.Linear(config.EMBED_SIZE, n_pois)
        
        # Gated Fusion Layer
        self.gate_layer = nn.Sequential(
            nn.Linear(config.EMBED_SIZE, 1),
            nn.Sigmoid()
        )
        
        self.register_buffer('flow_map', flow_map_tensor)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, batch):
        session = batch['session']; mask = batch['mask']; last_poi = batch['last_poi']
        batch_size, seq_len = session.shape
        
        # 1. Transformer Branch (Local Context)
        poi_embeds = self.poi_embedding(session)
        positions = torch.arange(seq_len, device=session.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)
        input_embeds = self.dropout(poi_embeds + pos_embeds)
        
        padding_mask = ~mask
        transformer_output = self.transformer_encoder(input_embeds, src_key_padding_mask=padding_mask)
        
        last_item_indices = mask.sum(dim=1) - 1
        h_local = transformer_output[torch.arange(batch_size), last_item_indices]
        
        # 局部预测分数
        logits_local = self.fc_local(h_local)

        # 2. Flow Map Branch (Global Context)
        # 从 flow_map 中查找 last_poi 对应的转移概率分布
        p_global = self.flow_map[last_poi] # (B, n_pois)
        
        # 将概率转换为 log-space scores
        logits_global = torch.log(p_global + 1e-8) # 加 epsilon 防止 log(0)

        # 3. Gated Fusion
        gate = self.gate_layer(h_local) # (B, 1)
        
        # 融合局部和全局预测
        final_logits = gate * logits_local + (1 - gate) * logits_global
        
        return final_logits

# --- 5. 训练和评估函数 ---
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        for k, v in batch.items(): batch[k] = v.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch['target'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, top_k, device):
    # This function is identical to the one in FPG-T and can be reused.
    # For brevity, I'll assume it's defined elsewhere. A copy is provided below if needed.
    model.eval()
    metrics = {f'acc@{k}': 0 for k in top_k}; metrics['mrr'] = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for k, v in batch.items(): batch[k] = v.to(device)
            logits = model(batch); target = batch['target']
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
    for k in top_k: metrics[f'acc@{k}'] /= total_samples
    metrics['mrr'] /= total_samples
    return metrics

# --- 6. 主程序 ---
if __name__ == '__main__':
    config = Config()
    print(f"Using device: {config.DEVICE}")

    # 1. PREPROCESS DATA
    train_data, valid_data, test_data, n_pois, flow_map_tensor = preprocess_data_for_getnext(config)
    print(f"\nNum POIs: {n_pois}")
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    # 2. CREATE DATALOADERS
    train_loader = DataLoader(GETNextDataset(train_data), batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(GETNextDataset(valid_data), batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(GETNextDataset(test_data), batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. INITIALIZE MODEL & CO.
    model = GETNext(n_pois, flow_map_tensor, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 4. TRAINING LOOP
    best_val_mrr = 0.0; best_model_state = None; patience = 5; patience_counter = 0
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        avg_loss = train_model(model, train_loader, optimizer, criterion, config.DEVICE)
        print(f"Avg Train Loss: {avg_loss:.4f}")
        
        val_metrics = evaluate(model, valid_loader, config.TOP_K, config.DEVICE)
        print("--- Validation Metrics ---")
        for metric, value in val_metrics.items(): print(f"{metric}: {value:.4f}")
        
        current_val_mrr = val_metrics['mrr']
        if current_val_mrr > best_val_mrr:
            best_val_mrr = current_val_mrr; best_model_state = model.state_dict(); patience_counter = 0
            print("Validation MRR improved! Saving model.")
        else:
            patience_counter += 1; print(f"No improvement for {patience_counter} epoch(s).")
        if patience_counter >= patience:
            print("Early stopping triggered."); break

    # 5. FINAL EVALUATION
    print("\n--- Training Finished ---")
    if best_model_state:
        print("Loading best model for final testing...")
        model.load_state_dict(best_model_state)
    
    print("\n--- Final Test Metrics ---")
    test_metrics = evaluate(model, test_loader, config.TOP_K, config.DEVICE)
    for metric, value in test_metrics.items(): print(f"{metric}: {value:.4f}")