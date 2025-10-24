import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

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
    N_HEADS = 4          # Transformer 注意力头数
    N_LAYERS = 2         # Transformer Encoder 层数
    DROPOUT = 0.3
    
    # 训练参数
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 30
    WEIGHT_DECAY = 1e-5

    # 数据处理参数
    SESSION_LEN = 20         # 考虑的最大历史序列长度
    MIN_TRAJ_LEN = 3         # 用户的最短轨迹长度

    # 评估参数
    TOP_K = [1, 5, 10]

# --- 2. 数据预处理 ---
def preprocess_data_for_fpgt(config):
    print("Step 1: Loading and mapping data...")
    df = pd.read_csv(config.DATA_PATH)
    df.rename(columns={'geo_id': 'poi_id', 'venue_category_id': 'category_id'}, inplace=True)
    
    # --- 创建三个核心映射 ---
    # 1. POI ID 映射 (0 for padding)
    poi_map = {poi: i + 1 for i, poi in enumerate(df['poi_id'].unique())}
    n_pois = len(poi_map) + 1
    
    # 2. Category ID 映射 (0 for padding/unknown)
    category_map = {cat: i + 1 for i, cat in enumerate(df['category_id'].unique())}
    n_categories = len(category_map) + 1
    
    # 3. User ID 映射
    user_map = {user: i for i, user in enumerate(df['user_id'].unique())}
    n_users = len(user_map)

    # 应用映射
    df['poi_id_mapped'] = df['poi_id'].map(poi_map)
    df['category_id_mapped'] = df['category_id'].map(category_map)
    df['user_id_mapped'] = df['user_id'].map(user_map)

    # 创建一个 POI -> Category 的查找表
    poi_to_category = df.drop_duplicates(subset=['poi_id_mapped'])[['poi_id_mapped', 'category_id_mapped']]
    poi_to_category_map = torch.zeros(n_pois, dtype=torch.long)
    for _, row in poi_to_category.iterrows():
        poi_to_category_map[int(row['poi_id_mapped'])] = int(row['category_id_mapped'])
    
    # 按用户和时间排序
    df.sort_values(by=['user_id_mapped', 'time'], inplace=True)

    # --- 生成样本并按 8:1:1 划分 ---
    print("Step 2: Generating samples and splitting (8:1:1)...")
    train_data, valid_data, test_data = [], [], []
    
    user_groups = df.groupby('user_id_mapped')
    for user_id, group in tqdm(user_groups, desc="Processing users"):
        traj = group['poi_id_mapped'].tolist()
        
        num_checkins = len(traj)
        if num_checkins < config.MIN_TRAJ_LEN:
            continue
            
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
        
    return train_data, valid_data, test_data, n_users, n_pois, n_categories, poi_to_category_map

# --- 3. PyTorch Dataset 和 DataLoader ---
class FPGTDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'session': torch.tensor(item['session'], dtype=torch.long),
            'target': torch.tensor(item['target'], dtype=torch.long)
        }

def collate_fn(batch):
    sessions = [item['session'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    padded_sessions = pad_sequence(sessions, batch_first=True, padding_value=0)
    mask = (padded_sessions != 0)
    return {'session': padded_sessions, 'mask': mask, 'target': targets}

# --- 4. FPG-T 模型定义 ---
class FPGT(nn.Module):
    def __init__(self, n_pois, n_categories, poi_to_category_map, config):
        super(FPGT, self).__init__()
        self.config = config
        
        # 嵌入层
        self.poi_embedding = nn.Embedding(n_pois, config.EMBED_SIZE, padding_idx=0)
        self.category_embedding = nn.Embedding(n_categories, config.EMBED_SIZE, padding_idx=0)
        self.pos_embedding = nn.Embedding(config.SESSION_LEN, config.EMBED_SIZE)
        
        # 将 POI->Category 映射表注册为 buffer
        self.register_buffer('poi_to_category_map', poi_to_category_map)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_SIZE,
            nhead=config.N_HEADS,
            dim_feedforward=config.EMBED_SIZE * 4,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.N_LAYERS)
        
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(config.EMBED_SIZE, n_pois)

    def forward(self, batch):
        session = batch['session'] # (B, S)
        mask = batch['mask']       # (B, S)
        
        batch_size, seq_len = session.shape
        
        # 1. 获取 POI 嵌入
        poi_embeds = self.poi_embedding(session)
        
        # 2. 获取 Category (Group) 嵌入
        category_ids = self.poi_to_category_map[session]
        category_embeds = self.category_embedding(category_ids)
        
        # 3. 组合嵌入 (核心思想)
        input_embeds = poi_embeds + category_embeds
        
        # 4. 添加位置编码
        positions = torch.arange(seq_len, device=session.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)
        input_embeds = self.dropout(input_embeds + pos_embeds)
        
        # 5. 通过 Transformer
        # PyTorch Transformer's padding mask needs True for padded items
        padding_mask = ~mask
        transformer_output = self.transformer_encoder(input_embeds, src_key_padding_mask=padding_mask)
        
        # 6. 提取最后一个有效 POI 的表示
        last_item_indices = mask.sum(dim=1) - 1
        last_item_reps = transformer_output[torch.arange(batch_size), last_item_indices]
        
        # 7. 预测
        logits = self.fc(last_item_reps)
        return logits

# --- 5. 训练和评估函数 (可复用) ---
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
    train_data, valid_data, test_data, n_users, n_pois, n_categories, poi_to_category_map = preprocess_data_for_fpgt(config)
    print(f"\nNum users: {n_users}, Num POIs: {n_pois}, Num Categories: {n_categories}")
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    # 2. CREATE DATALOADERS
    train_loader = DataLoader(FPGTDataset(train_data), batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(FPGTDataset(valid_data), batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(FPGTDataset(test_data), batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. INITIALIZE MODEL & CO.
    model = FPGT(n_pois, n_categories, poi_to_category_map, config).to(config.DEVICE)
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