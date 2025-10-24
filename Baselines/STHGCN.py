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
    # 数据路径
    BASE_DIR = os.getcwd()
    DATA_PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_tky_cleaned.csv')

    # 模型参数
    EMBED_SIZE = 128
    ATTN_HEADS = 4
    DROPOUT = 0.3
    
    # 训练参数
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 30
    WEIGHT_DECAY = 1e-5

    # 数据处理参数
    SESSION_LEN = 10         # 定义一个会话（用于构建超图）的最大长度
    MIN_TRAJ_LEN = 3         # 用户的最短轨迹长度

    # 评估参数
    TOP_K = [1, 5, 10]

# --- 2. 数据预处理 (针对 Session-based 模型) ---
def preprocess_data_for_session_hg(config):
    print("Step 1: Loading and mapping data...")
    df = pd.read_csv(config.DATA_PATH)
    df.rename(columns={'geo_id': 'poi_id'}, inplace=True)
    
    poi_map = {poi: i + 1 for i, poi in enumerate(df['poi_id'].unique())}
    n_pois = len(poi_map) + 1
    
    df['poi_id_mapped'] = df['poi_id'].map(poi_map).fillna(0).astype(int)
    
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['user_id', 'time'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("Step 2: Generating session-based data and splitting (8:1:1)...")
    train_data, valid_data, test_data = [], [], []
    
    user_groups = df.groupby('user_id')
    for user_id, group in tqdm(user_groups, desc="Processing users"):
        traj = group['poi_id_mapped'].tolist()
        
        num_checkins = len(traj)
        if num_checkins < config.MIN_TRAJ_LEN or num_checkins < 5:
            continue
            
        # Calculate split indices
        train_end_idx = int(num_checkins * 0.8)
        valid_end_idx = int(num_checkins * 0.9)
            
        # Create sessions and assign them to datasets
        for i in range(1, num_checkins):
            start_idx = max(0, i - config.SESSION_LEN)
            session = traj[start_idx:i]
            target = traj[i]
            
            # Skip empty sessions
            if not session:
                continue

            sample = {'session': session, 'target': target}
            
            # Assign sample based on the target's position in the full trajectory
            if i < train_end_idx:
                train_data.append(sample)
            elif i < valid_end_idx:
                valid_data.append(sample)
            else:
                test_data.append(sample)
        
    return train_data, valid_data, test_data, n_pois

# --- 3. PyTorch Dataset 和 DataLoader ---
class SessionDataset(Dataset):
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
    
    # 对 session 进行 padding
    padded_sessions = pad_sequence(sessions, batch_first=True, padding_value=0)
    
    # 创建 mask
    mask = (padded_sessions != 0)
    
    return {
        'session': padded_sessions,
        'mask': mask,
        'target': targets
    }

# --- 4. STHGL 模型定义 ---
class HypergraphAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super(HypergraphAttention, self).__init__()
        assert embed_size % num_heads == 0
        self.d_k = embed_size // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.W_o = nn.Linear(embed_size, embed_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, embed_size)
        # mask: (batch_size, seq_len)
        
        batch_size, seq_len, _ = x.shape
        
        # 1. 线性变换
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力分数
        # (batch, heads, seq_len, d_k) @ (batch, heads, d_k, seq_len) -> (batch, heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. 应用 mask (防止注意到 padding item)
        mask = mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 4. 加权求和
        context = torch.matmul(attn_probs, v) # (batch, heads, seq_len, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 5. 输出线性变换
        output = self.W_o(context)
        return output

class STHGL(nn.Module):
    def __init__(self, n_pois, config):
        super(STHGL, self).__init__()
        self.n_pois = n_pois
        self.embed_size = config.EMBED_SIZE

        # 嵌入层
        self.poi_embedding = nn.Embedding(n_pois, self.embed_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(config.SESSION_LEN, self.embed_size) # 位置嵌入

        # 核心模块
        self.hypergraph_attention = HypergraphAttention(
            embed_size=self.embed_size,
            num_heads=config.ATTN_HEADS,
            dropout=config.DROPOUT
        )
        
        self.layer_norm = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # 预测层
        self.fc = nn.Linear(self.embed_size, n_pois)

    def forward(self, batch):
        session = batch['session'] # (batch_size, seq_len)
        mask = batch['mask']       # (batch_size, seq_len)
        
        batch_size, seq_len = session.shape
        
        # 1. 获取嵌入
        poi_embeds = self.poi_embedding(session)
        
        # 2. 添加位置编码
        positions = torch.arange(seq_len, device=session.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)
        x = poi_embeds + pos_embeds
        x = self.dropout(x)
        
        # 3. 通过超图注意力层
        # 在这篇论文中，整个会话被看作一个超图，注意力机制在会话内的节点间进行
        attn_output = self.hypergraph_attention(x, mask)
        
        # 4. 残差连接和层归一化
        x = self.layer_norm(x + attn_output)
        
        # 5. 提取最后一个非 padding POI 的表示用于预测
        # 这是会话推荐模型的标准做法
        # seq_len - 1 is not correct due to padding
        last_item_indices = mask.sum(dim=1) - 1 # (batch_size)
        # Gather the last hidden states
        last_item_reps = x[torch.arange(batch_size), last_item_indices] # (batch_size, embed_size)
        
        # 6. 预测
        logits = self.fc(last_item_reps)
        
        return logits

# --- 5. 训练和评估函数 ---
def train_session_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        # Move data to device
        for k, v in batch.items():
            batch[k] = v.to(device)

        optimizer.zero_grad()
        logits = model(batch)
        
        target = batch['target'].to(device)
        loss = criterion(logits, target)
        
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
        for batch in tqdm(dataloader, desc="Evaluating"):
            for k, v in batch.items():
                batch[k] = v.to(device)

            logits = model(batch)
            target = batch['target']
            
            max_k = max(top_k)
            _, top_indices = torch.topk(logits, max_k, dim=1)
            expanded_target = target.view(-1, 1).expand_as(top_indices)
            hits_matrix = (top_indices == expanded_target)

            for k in top_k:
                hits_in_k = hits_matrix[:, :k].any(dim=1).sum().item()
                metrics[f'acc@{k}'] += hits_in_k
            
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

# --- 6. 主程序 ---
if __name__ == '__main__':
    config = Config()
    print(f"Using device: {config.DEVICE}")

    # 1. PREPROCESS DATA
    train_data, valid_data, test_data, n_pois = preprocess_data_for_session_hg(config)
    print(f"\nNumber of POIs: {n_pois}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    print(f"Testing samples: {len(test_data)}")

    # 2. CREATE DATALOADERS
    train_dataset = SessionDataset(train_data)
    valid_dataset = SessionDataset(valid_data)
    test_dataset = SessionDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. INITIALIZE MODEL & CO.
    model = STHGL(n_pois, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 4. TRAINING LOOP
    best_val_mrr = 0.0
    best_model_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        avg_train_loss = train_session_model(model, train_loader, optimizer, criterion, config.DEVICE)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Evaluate on validation set
        val_metrics = evaluate(model, valid_loader, config.TOP_K, config.DEVICE)
        print("--- Validation Metrics ---")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Early stopping logic
        current_val_mrr = val_metrics['mrr']
        if current_val_mrr > best_val_mrr:
            best_val_mrr = current_val_mrr
            best_model_state = model.state_dict()
            patience_counter = 0
            print("Validation MRR improved! Saving model state.")
        else:
            patience_counter += 1
            print(f"No improvement in validation MRR for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs.")
            break

    # 5. FINAL EVALUATION
    print("\n--- Training Finished ---")
    if best_model_state:
        print("Loading best model for final testing...")
        model.load_state_dict(best_model_state)
    
    print("\n--- Final Test Metrics ---")
    test_metrics = evaluate(model, test_loader, config.TOP_K, config.DEVICE)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")