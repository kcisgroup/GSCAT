import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- 1. 配置参数 (Hyperparameters & Configuration) ---
class Config:
    # 数据路径
    BASE_DIR = os.getcwd()
    DATA_PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_nyc_cleaned.csv')

    # 模型参数
    EMBED_SIZE = 128         # 用户、POI、小时、星期的嵌入维度
    HIDDEN_SIZE = 128        # GRU 隐藏层维度
    NUM_LAYERS = 1           # GRU 层数
    DROPOUT = 0.3            # Dropout 比率

    # 训练参数
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 20
    MAX_SEQ_LEN = 20         # 考虑的最大历史序列长度
    MIN_SEQ_LEN = 3          # 一个用户轨迹的最短长度，少于此长度的将被丢弃

    # 评估参数
    TOP_K = [1, 5, 10]       # 计算 Accuracy@1, @5, @10

# --- 2. 数据预处理 ---
def preprocess_data(config):
    """
    加载数据、创建ID映射、生成用户轨迹序列 (按8:1:1划分)
    """
    print("Step 1: Loading and preprocessing data...")
    df = pd.read_csv(config.DATA_PATH)

    df.rename(columns={'geo_id': 'poi_id'}, inplace=True)
    
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['user_id', 'time'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    poi_map = {poi: i + 1 for i, poi in enumerate(df['poi_id'].unique())}
    user_map = {user: i for i, user in enumerate(df['user_id'].unique())}
    
    n_pois = len(poi_map) + 1
    n_users = len(user_map)
    
    df['poi_id_mapped'] = df['poi_id'].map(poi_map)
    df['user_id_mapped'] = df['user_id'].map(user_map)

    print("Step 2: Generating user sequences and splitting (8:1:1)...")
    train_data, valid_data, test_data = [], [], []
    
    user_groups = df.groupby('user_id_mapped')
    for user, group in tqdm(user_groups, desc="Processing users"):
        pois = group['poi_id_mapped'].tolist()
        hours = group['hour'].tolist()
        weekdays = group['weekday'].tolist()
        
        num_checkins = len(pois)
        if num_checkins < config.MIN_SEQ_LEN or num_checkins < 5:
            continue
            
        # 计算划分点
        train_end_idx = int(num_checkins * 0.8)
        valid_end_idx = int(num_checkins * 0.9)

        # 遍历序列，创建 (input_sequence, target_poi) 对
        for i in range(1, num_checkins):
            start_idx = max(0, i - config.MAX_SEQ_LEN)
            
            sample = {
                'user': user,
                'poi_seq': pois[start_idx:i],
                'hour_seq': hours[start_idx:i],
                'weekday_seq': weekdays[start_idx:i],
                'target_poi': pois[i]
            }
            
            # 根据目标点的位置来决定样本属于哪个集合
            if i < train_end_idx:
                train_data.append(sample)
            elif i < valid_end_idx:
                valid_data.append(sample)
            else:
                test_data.append(sample)

    return train_data, valid_data, test_data, n_users, n_pois

# --- 3. PyTorch Dataset 和 DataLoader ---
class POIDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'user': torch.tensor(item['user'], dtype=torch.long),
            'poi_seq': torch.tensor(item['poi_seq'], dtype=torch.long),
            'hour_seq': torch.tensor(item['hour_seq'], dtype=torch.long),
            'weekday_seq': torch.tensor(item['weekday_seq'], dtype=torch.long),
            'target_poi': torch.tensor(item['target_poi'], dtype=torch.long)
        }

def collate_fn(batch):
    # 自定义 collate_fn 来处理变长序列的 padding
    batch_dict = {
        'user': torch.stack([item['user'] for item in batch]),
        'target_poi': torch.stack([item['target_poi'] for item in batch]),
        'seq_len': torch.tensor([len(item['poi_seq']) for item in batch], dtype=torch.long)
    }
    
    # 使用 pad_sequence，它需要一个 list of tensors
    # padding_value=0, 因为我们在映射时为 padding 预留了 0
    batch_dict['poi_seq'] = pad_sequence([item['poi_seq'] for item in batch], batch_first=True, padding_value=0)
    batch_dict['hour_seq'] = pad_sequence([item['hour_seq'] for item in batch], batch_first=True, padding_value=0)
    batch_dict['weekday_seq'] = pad_sequence([item['weekday_seq'] for item in batch], batch_first=True, padding_value=0)
    
    return batch_dict

# --- 4. GLCIF 模型定义 ---
class GLCIF(nn.Module):
    def __init__(self, n_users, n_pois, config):
        super(GLCIF, self).__init__()
        self.n_users = n_users
        self.n_pois = n_pois
        self.embed_size = config.EMBED_SIZE
        self.hidden_size = config.HIDDEN_SIZE

        # Embedding Layers
        self.user_embed = nn.Embedding(n_users, self.embed_size)
        self.poi_embed = nn.Embedding(n_pois, self.embed_size, padding_idx=0) # padding_idx=0 表示 0 不参与梯度更新
        self.hour_embed = nn.Embedding(24, self.embed_size)
        self.weekday_embed = nn.Embedding(7, self.embed_size)
        
        # Local Context Module (短期序列)
        # 输入维度是 POI embedding + hour embedding + weekday embedding
        input_gru_size = self.embed_size * 3 
        self.gru = nn.GRU(input_gru_size, self.hidden_size, config.NUM_LAYERS, batch_first=True, dropout=config.DROPOUT)
        
        # Fusion Gate
        self.fusion_gate_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size), # hidden_size for local, hidden_size for global
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Prediction Layer
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(self.hidden_size, self.n_pois)

    def forward(self, batch):
        # --- a. 获取输入 ---
        user = batch['user']             # (batch_size)
        poi_seq = batch['poi_seq']       # (batch_size, seq_len)
        hour_seq = batch['hour_seq']     # (batch_size, seq_len)
        weekday_seq = batch['weekday_seq'] # (batch_size, seq_len)
        seq_len = batch['seq_len']       # (batch_size)

        # --- b. 全局上下文 (Global Context) ---
        # 用户的长期偏好，直接用 user embedding 表示
        h_global = self.user_embed(user) # (batch_size, embed_size)
        
        # --- c. 局部上下文 (Local Context) ---
        # 获取序列中每个元素的嵌入
        poi_seq_embed = self.poi_embed(poi_seq)
        hour_seq_embed = self.hour_embed(hour_seq)
        weekday_seq_embed = self.weekday_embed(weekday_seq)

        # 将 POI 和上下文特征拼接
        seq_embed = torch.cat([poi_seq_embed, hour_seq_embed, weekday_seq_embed], dim=-1) # (batch_size, seq_len, embed_size * 3)

        # 为了处理padding，使用 pack_padded_sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(seq_embed, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed_input)
        
        # h_n 的 shape 是 (num_layers, batch_size, hidden_size)，我们取最后一层
        h_local = h_n.squeeze(0) # (batch_size, hidden_size)

        # --- d. 信息融合 (Information Fusion) ---
        # 拼接全局和局部上下文
        combined_context = torch.cat([h_global, h_local], dim=-1)
        
        # 计算融合门
        gate = self.fusion_gate_layer(combined_context) # (batch_size, 1)
        
        # 动态融合
        h_final = gate * h_global + (1 - gate) * h_local # (batch_size, hidden_size)

        # --- e. 预测 ---
        h_final = self.dropout(h_final)
        logits = self.fc(h_final) # (batch_size, n_pois)
        
        return logits

# --- 5. 训练和评估函数  ---

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        # Move data to device
        for k, v in batch.items():
            batch[k] = v.to(device)

        optimizer.zero_grad()
        logits = model(batch)
        
        # 注意 target_poi 也需要移动到 device
        target = batch['target_poi'].to(device)
        loss = criterion(logits, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, top_k, device):
    """
    评估函数，包含 Acc@k 和 MRR
    """
    model.eval()
    # 初始化指标字典
    metrics = {f'acc@{k}': 0 for k in top_k}
    metrics['mrr'] = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 将数据移动到指定设备
            for k, v in batch.items():
                batch[k] = v.to(device)
                
            logits = model(batch)
            target = batch['target_poi']

            # 获取 Top K 预测，K 取最大值
            max_k = max(top_k)
            _, top_indices = torch.topk(logits, max_k, dim=1) # (batch_size, max_k)

            # 将目标形状扩展以便于比较
            # target: (batch_size) -> (batch_size, 1) -> (batch_size, max_k)
            expanded_target = target.view(-1, 1).expand_as(top_indices)

            # --- 核心计算 ---
            # 1. 创建一个布尔矩阵，表示预测是否命中目标
            hits_matrix = (top_indices == expanded_target)

            # 2. 计算 Acc@k
            for k in top_k:
                # 检查前 k 个预测中是否有命中 (True)
                # .any(dim=1) 会检查每一行是否有 True
                # .sum() 统计命中的样本数
                hits_in_k = hits_matrix[:, :k].any(dim=1).sum().item()
                metrics[f'acc@{k}'] += hits_in_k
            
            # 3. 计算 MRR
            # torch.nonzero 找到所有 True 元素的位置
            # match_indices[0] 是行索引 (样本索引)
            # match_indices[1] 是列索引 (0-based 排名)
            match_indices = torch.nonzero(hits_matrix, as_tuple=True)
            
            # 初始化一个批次内所有样本的倒数排名张量
            reciprocal_ranks = torch.zeros(target.size(0)).to(device)
            
            # 计算命中样本的倒数排名 (排名是 1-based, 所以要 +1)
            ranks = match_indices[1].float() + 1.0
            
            # 使用 scatter_ 将计算出的倒数排名填充到对应样本的位置
            # 对于没有命中的样本，其值将保持为 0
            reciprocal_ranks.scatter_(0, match_indices[0], 1.0 / ranks)
            
            metrics['mrr'] += reciprocal_ranks.sum().item()
                
            total_samples += target.size(0)

    # 计算最终的平均指标
    for k in top_k:
        metrics[f'acc@{k}'] /= total_samples
    metrics['mrr'] /= total_samples
        
    return metrics

# --- 6. 主程序 ---
if __name__ == '__main__':
    config = Config()
    print(f"Using device: {config.DEVICE}")

    # 1. PREPROCESS DATA
    train_data, valid_data, test_data, n_users, n_pois = preprocess_data(config)
    print(f"\nNumber of users: {n_users}, Number of POIs: {n_pois}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    print(f"Testing samples: {len(test_data)}")

    # 2. CREATE DATALOADERS
    train_dataset = POIDataset(train_data)
    valid_dataset = POIDataset(valid_data)
    test_dataset = POIDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. INITIALIZE MODEL & CO.
    model = GLCIF(n_users, n_pois, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 4. TRAINING LOOP
    best_val_mrr = 0.0
    best_model_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        avg_train_loss = train(model, train_loader, optimizer, criterion, config.DEVICE)
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