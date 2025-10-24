import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import time
import copy
from functools import partial
import os
import optuna
from collections import defaultdict
from typing import Tuple

# --- PyTorch Geometric Imports ---
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pprint import pprint

# ==============================================================================
# 0. 配置项与全局变量
# ==============================================================================
# --- 路径配置 ---
BASE_DIR = os.getcwd()  # 获取当前工作目录
DATA_PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_nyc_cleaned.csv')  # 清洗后的签到数据路径
DISTANCE_DATA_PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/distance_df_nyc.csv')  # 预计算的POI间距离数据路径
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'MyModel/GSCAT/NYC/OptunaRuns')  # Optuna运行和模型保存目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)  # 确保目录存在
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'transformer_poi_cate_model_best_optuna.pth')  # 最佳模型保存路径
PRETRAINED_EMBED_DIR = os.path.join(BASE_DIR, 'MyModel/GSCAT/NYC/PretrainedEmbeddings')  # 预训练嵌入保存目录
os.makedirs(PRETRAINED_EMBED_DIR, exist_ok=True)
POI_EMBED_PATH = os.path.join(PRETRAINED_EMBED_DIR, 'poi_embeddings_transe.pt')
CAT_EMBED_PATH = os.path.join(PRETRAINED_EMBED_DIR, 'cat_embeddings_transe.pt')
USER_EMBED_PATH = os.path.join(PRETRAINED_EMBED_DIR, 'user_embeddings_transe.pt')

# --- TransE 预训练配置 ---
DO_TRANSE_PRETRAINING = True  # 是否执行或加载TransE预训练
TRANSE_EMBED_DIM = 128  # TransE嵌入维度
TRANSE_EPOCHS = 50  # TransE训练轮数
TRANSE_BATCH_SIZE = 4096  # TransE训练批次大小
TRANSE_LR = 0.001  # TransE学习率
TRANSE_MARGIN = 1.0  # TransE损失函数的margin
TRANSE_NEG_RATIO = 10  # 每个正样本对应的负采样数量

# --- 多关系图配置 ---
GLOBAL_GRAPH_DISTANCE_THRESHOLD = 2.5  # (km) 构建地理关系边的距离阈值
GLOBAL_GRAPH_COOCCUR_MIN_USERS = 3      # 构建共现关系边的最少共现用户数
GLOBAL_GRAPH_TOP_POPULAR_RATIO = 0.1    # 用于构建热门关系的POI比例
GLOBAL_GRAPH_SAME_CATEGORY_ENABLED = True  # 是否构建同类别关系边

# --- 关系类型定义 ---
RELATION_TYPES = {
    'GEOGRAPHICAL': 0,    # 地理相邻
    'SAME_CATEGORY': 1,   # 同类别
    'CO_OCCURRENCE': 2,   # 共现
    'POPULARITY': 3       # 热门POI连接
}
NUM_RELATION_TYPES = len(RELATION_TYPES)  # 关系总数

# --- 数据参数 ---
MAX_SEQ_LENGTH = 50  # 输入序列的最大长度
VAL_USER_RATIO = 0.1  # 验证集用户比例
TEST_USER_RATIO = 0.1 # 测试集用户比例
DATA_AUG_MASKING_RATIO = 0.1 # 训练时对输入序列进行随机masking的比例，一种数据增强手段

# --- 时间段语义类型定义 ---
# 定义不同时间段的规则，用于将连续时间映射为离散的语义类别
TIME_SEGMENT_RULES = [
    ("WORKDAY_MORNING_PEAK", lambda wd,h:0<=wd<=4 and 7<=h<=9),      # 工作日早高峰
    ("WORKDAY_DAYTIME",lambda wd,h:0<=wd<=4 and 10<=h<=16),     # 工作日白天
    ("WORKDAY_EVENING_PEAK",lambda wd,h:0<=wd<=4 and 17<=h<=19),     # 工作日晚高峰
    ("WORKDAY_NIGHT",lambda wd,h:0<=wd<=4 and (20<=h<=23 or 0<=h<=1)),# 工作日夜晚
    ("WEEKEND_DAYTIME",lambda wd,h:5<=wd<=6 and 9<=h<=17),      # 周末白天
    ("WEEKEND_NIGHT",lambda wd,h:5<=wd<=6 and (18<=h<=23 or 0<=h<=1)),# 周末夜晚
    ("LATE_NIGHT_DEEP",lambda wd,h:2<=h<=6)                       # 深夜
]
TIME_SEGMENT_CATEGORIES = {name: i for i,(name,_) in enumerate(TIME_SEGMENT_RULES)}
NUM_TIME_SEGMENTS = len(TIME_SEGMENT_CATEGORIES)
TIME_SEGMENT_PAD_IDX = NUM_TIME_SEGMENTS  # 时间段类别的填充索引
NUM_TIME_SEGMENTS_W_PAD = NUM_TIME_SEGMENTS + 1  # 加上填充后的总类别数

# --- 图边属性：时间差分箱 (用于GAT) ---
# 将连续的时间差（分钟）离散化为整数类别
EDGE_TIME_DIFF_BINS_GAT = [-1, 0, 5, 30, 120, 360, 1440, 1440*3, 1440*7, np.inf]
NUM_EDGE_TIME_BINS_GAT = len(EDGE_TIME_DIFF_BINS_GAT) - 1
EDGE_TIME_BIN_PAD_IDX_GAT = NUM_EDGE_TIME_BINS_GAT
NUM_EDGE_TIME_BINS_W_PAD_GAT = NUM_EDGE_TIME_BINS_GAT + 1

# --- 图边属性：地理距离分箱 (用于GAT) ---
# 将连续的地理距离（km）离散化为整数类别
EDGE_DIST_BINS_GAT = [0, 0.1, 0.5, 1, 2, 5, 10, 20, np.inf]
NUM_EDGE_DIST_BINS_GAT = len(EDGE_DIST_BINS_GAT) - 1
EDGE_DIST_BIN_PAD_IDX_GAT = NUM_EDGE_DIST_BINS_GAT
NUM_EDGE_DIST_BINS_W_PAD_GAT = NUM_EDGE_DIST_BINS_GAT + 1

# --- Transformer内部成对时间差分箱 ---
# 用于Transformer内部的Time-aware Attention，将序列中任意两点间的时间差（分钟）离散化
PAIRWISE_TIME_DIFF_BINS = [-float('inf'), -1440*7, -1440, -360, -120, -30, -5, 0, 5, 30, 120, 360, 1440, 1440*7, float('inf')]
NUM_PAIRWISE_TIME_DIFF_BINS = len(PAIRWISE_TIME_DIFF_BINS) - 1

# --- Optuna 和训练参数 ---
BATCH_SIZE = 128  # 批处理大小
EPOCHS_PER_TRIAL = 20  # Optuna每个试验的最大训练轮数
OPTUNA_N_TRIALS = 0  # Optuna要运行的总试验次数 (0表示只运行最终训练，不进行搜索)
PATIENCE = 5  # 早停的耐心轮数
WARMUP_STEPS = 1000  # 学习率预热的步数

DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")  # 设置计算设备
SEED = 42  # 随机种子，保证可复现性

# --- GAT 及 UserRepModule 相关配置的默认值 ---
# 这些值在Optuna搜索时可能被覆盖，但为非搜索运行提供基础配置
DEFAULT_GAT_NODE_ID_EMBED_DIM = 32
DEFAULT_GAT_NODE_CAT_EMBED_DIM = 16
DEFAULT_GAT_NODE_LOC_EMBED_DIM = 16
DEFAULT_EDGE_TIME_EMBED_DIM_GAT = 8
DEFAULT_EDGE_DIST_EMBED_DIM_GAT = 8
DEFAULT_GAT_HIDDEN_DIMS = [64] 
DEFAULT_GAT_NUM_HEADS_LIST = [4]
DEFAULT_GAT_OUTPUT_DIM = 64 
DEFAULT_GAT_DROPOUT = 0.1
DEFAULT_USER_REP_TF_D_MODEL = 64
DEFAULT_USER_REP_FUSION_TYPE = 'concat_mlp'

# --- 全局多关系图配置 ---
DEFAULT_GLOBAL_GRAPH_HIDDEN_DIM = 128
DEFAULT_GLOBAL_GRAPH_NUM_LAYERS = 2
DEFAULT_GLOBAL_GRAPH_NUM_HEADS = 4
DEFAULT_GLOBAL_GRAPH_DROPOUT = 0.1

# --- 全局变量 ---
# 这些变量将在数据预处理阶段被赋值，并在整个脚本中共享
num_categories = 0; cat_pad_idx = 0; num_categories_with_pad = 0
num_venues = 0; venue_pad_idx = 0; num_venues_with_pad = 0
venue_map = {}; category_map = {}
global_distance_lookup = None # 全局距离查找表
global_all_train_user_reps_tensor = None # 存储所有训练用户表示的张量
global_train_user_id_to_idx_map = None # 训练用户ID到张量索引的映射
global_idx_to_train_user_id_map = None # 张量索引到训练用户ID的映射
global_train_user_ids_list_for_rep_calc = [] # 需要计算表示的训练用户ID列表
global_poi_graph_data = None # 全局POI图数据

# ==============================================================================
# 0.5. TransE 预训练模块
# ==============================================================================
class TransEDataset(Dataset):
    """
    用于TransE预训练的PyTorch数据集。
    
    该数据集为每个正三元组 (head, relation, tail) 生成指定数量的负三元组。
    负三元组通过随机替换原始尾实体来创建（尾部损坏）。
    """
    def __init__(self, triplets, num_entities, neg_ratio=1):
        """
        初始化TransE数据集。
        
        Args:
            triplets (list): 正三元组 (h, r, t) 的列表。
            num_entities (int): 知识图中实体的总数。
            neg_ratio (int): 每个正三元组生成的负三元组数量。
        """
        self.triplets = triplets
        self.num_entities = num_entities
        self.neg_ratio = neg_ratio
        self.entity_list = list(range(num_entities))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        获取一个样本，包含一个正三元组和多个负三元组。
        """
        pos_triplet = self.triplets[idx]
        h, r, t = pos_triplet
        
        neg_triplets = []
        for _ in range(self.neg_ratio):
            # 随机选择一个负尾实体，确保它不等于正尾实体
            neg_t = random.choice(self.entity_list)
            while neg_t == t:
                neg_t = random.choice(self.entity_list)
            neg_triplets.append((h, r, neg_t))
            
        return torch.LongTensor(pos_triplet), torch.LongTensor(neg_triplets)

class TransE(nn.Module):
    """
    TransE模型的PyTorch实现。
    
    TransE旨在通过将关系解释为实体嵌入空间中的平移来学习实体和关系的嵌入。
    理想情况下，对于一个三元组 (h, r, t)，嵌入应满足 h_emb + r_emb ≈ t_emb。
    """
    def __init__(self, num_entities, num_relations, embed_dim, margin=1.0, norm=2):
        """
        初始化TransE模型。
        
        Args:
            num_entities (int): 实体总数。
            num_relations (int): 关系总数。
            embed_dim (int): 嵌入向量的维度。
            margin (float): 边际排名损失中的边际值。
            norm (int): 用于计算距离的范数（通常是L1或L2）。
        """
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, embed_dim)
        self.relation_embedding = nn.Embedding(num_relations, embed_dim)
        self.margin = margin
        self.norm = norm
        
        # 使用Xavier均匀初始化嵌入权重
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def _get_embedding(self, h, r, t):
        """辅助函数，获取头、关系和尾的嵌入。"""
        h_emb = self.entity_embedding(h)
        r_emb = self.relation_embedding(r)
        t_emb = self.entity_embedding(t)
        return h_emb, r_emb, t_emb

    def forward(self, pos_triplets, neg_triplets):
        """
        计算TransE的边际排名损失。
        
        Args:
            pos_triplets (Tensor): 正三元组的批次。
            neg_triplets (Tensor): 负三元组的批次。
        
        Returns:
            Tensor: 计算出的损失值。
        """
        # 计算正三元组的分数（距离）
        h_pos, r_pos, t_pos = self._get_embedding(pos_triplets[:, 0], pos_triplets[:, 1], pos_triplets[:, 2])
        pos_scores = torch.norm(h_pos + r_pos - t_pos, p=self.norm, dim=1)

        # 计算负三元组的分数（距离）
        h_neg, r_neg, t_neg = self._get_embedding(neg_triplets[..., 0], neg_triplets[..., 1], neg_triplets[..., 2])
        neg_scores = torch.norm(h_neg + r_neg - t_neg, p=self.norm, dim=-1)

        # 计算边际排名损失
        loss = F.relu(self.margin + pos_scores.unsqueeze(1) - neg_scores).mean()
        return loss

def prepare_transe_data(df, user_map, venue_map, category_map):
    """
    从DataFrame准备用于TransE训练的三元组数据。
    
    它将用户、POI和类别映射到一个统一的实体空间，并创建两种关系的三元组：
    1. (user, 'user_attends_poi', poi)
    2. (poi, 'poi_is_category', category)
    
    Returns:
        tuple: 包含三元组列表、实体到ID的映射、关系到ID的映射、实体总数和关系总数。
    """
    print("准备TransE预训练数据...")
    num_users = len(user_map)
    num_pois = len(venue_map)
    num_categories = len(category_map)
    
    # 创建统一的实体空间和映射
    entity2id = {}
    for user_str_id, user_int_id in user_map.items():
        entity2id[f"u_{user_str_id}"] = user_int_id
    for poi_str_id, poi_int_id in venue_map.items():
        # POI的ID在用户ID之后
        entity2id[f"p_{poi_str_id}"] = num_users + poi_int_id
    for cat_str_id, cat_int_id in category_map.items():
        # 类别的ID在用户和POI的ID之后
        entity2id[f"c_{cat_str_id}"] = num_users + num_pois + cat_int_id
    num_entities = len(entity2id)
    print(f"TransE实体总数: {num_entities}")

    relation2id = {'user_attends_poi': 0, 'poi_is_category': 1}
    num_relations = len(relation2id)
    print(f"TransE关系总数: {num_relations}")

    triplets = []
    # 构建用户-POI三元组
    user_poi_pairs = df[['user_id', 'geo_id']].drop_duplicates()
    for _, row in tqdm(user_poi_pairs.iterrows(), total=len(user_poi_pairs), desc="构建 user-poi三元组"):
        h = entity2id.get(f"u_{row['user_id']}")
        t = entity2id.get(f"p_{row['geo_id']}")
        r = relation2id['user_attends_poi']
        if h is not None and t is not None:
            triplets.append((h, r, t))

    # 构建POI-类别三元组
    poi_cat_pairs = df[['geo_id', 'venue_category_id']].drop_duplicates()
    for _, row in tqdm(poi_cat_pairs.iterrows(), total=len(poi_cat_pairs), desc="构建 poi-category三元组"):
        cat_id_str = str(row['venue_category_id'])
        h = entity2id.get(f"p_{row['geo_id']}")
        t = entity2id.get(f"c_{cat_id_str}")
        r = relation2id['poi_is_category']
        if h is not None and t is not None:
            triplets.append((h, r, t))
            
    print(f"构建完成，总三元组数量: {len(triplets)}")
    return triplets, entity2id, relation2id, num_entities, num_relations

def run_transe_pretraining(triplets, num_entities, num_relations, embed_dim, device,
                           epochs, batch_size, lr, margin, neg_ratio):
    """
    执行TransE模型的训练循环。
    
    Args:
        triplets (list): 用于训练的三元组列表。
        ... (其他参数): TransE模型和训练过程的超参数。
    
    Returns:
        Tensor: 训练完成的实体嵌入权重。
    """
    print("\n--- 开始 TransE 预训练 ---")
    dataset = TransEDataset(triplets, num_entities, neg_ratio=neg_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = TransE(num_entities, num_relations, embed_dim, margin=margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"TransE Epoch {epoch}/{epochs}", leave=False)
        for pos_batch, neg_batch in pbar:
            pos_batch, neg_batch = pos_batch.to(device), neg_batch.to(device)
            optimizer.zero_grad()
            loss = model(pos_batch, neg_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"TransE Epoch {epoch} 完成, 平均损失: {avg_loss:.4f}")

    print("--- TransE 预训练结束 ---")
    return model.entity_embedding.weight.data.detach().cpu()

def extract_and_save_embeddings(pre_trained_embeddings, user_map, venue_map, category_map,
                                num_users, num_venues_with_pad, num_categories_with_pad, 
                                transe_embed_dim, user_path, poi_path, cat_path):
    """
    从统一的TransE实体嵌入中提取特定于用户、POI和类别的嵌入，并保存它们。
    
    Args:
        pre_trained_embeddings (Tensor): TransE训练得到的完整实体嵌入。
        ... (其他参数): 各种映射和维度信息。
    
    Returns:
        tuple: 包含用户、POI和类别嵌入张量的元组。
    """
    print("从TransE结果中提取并保存用户、POI、类别嵌入...")
    num_pois_map = len(venue_map)

    user_embeddings = torch.zeros(len(user_map), transe_embed_dim)
    for user_str_id, user_int_id in user_map.items():
        user_embeddings[user_int_id] = pre_trained_embeddings[user_int_id]

    poi_embeddings = torch.zeros(num_venues_with_pad, transe_embed_dim)
    for poi_str_id, poi_int_id in venue_map.items():
        transe_poi_id = num_users + poi_int_id
        poi_embeddings[poi_int_id] = pre_trained_embeddings[transe_poi_id]

    category_embeddings = torch.zeros(num_categories_with_pad, transe_embed_dim)
    for cat_str_id, cat_int_id in category_map.items():
        transe_cat_id = num_users + num_pois_map + cat_int_id
        category_embeddings[cat_int_id] = pre_trained_embeddings[transe_cat_id]
        
    torch.save(user_embeddings, user_path)
    torch.save(poi_embeddings, poi_path)
    torch.save(category_embeddings, cat_path)
    print(f"嵌入已保存到: {os.path.dirname(user_path)}")
    return user_embeddings, poi_embeddings, category_embeddings

# ==============================================================================
# 0.6. 多关系全局POI图构建模块
# ==============================================================================
class GlobalPOIGraphBuilder:
    """
    一个用于构建多关系全局POI图的工具类。
    
    该图包含多种类型的边，代表POI之间的不同关系，例如地理邻近性、类别相似性、共现性和热门度。
    """
    
    def __init__(self, venue_map, category_map, distance_lookup, df):
        """
        初始化图构建器。
        
        Args:
            venue_map (dict): POI ID到整数索引的映射。
            category_map (dict): 类别ID到整数索引的映射。
            distance_lookup (dict): 预计算的POI间距离查找表。
            df (pd.DataFrame): 原始签到数据。
        """
        self.venue_map = venue_map
        self.category_map = category_map 
        self.distance_lookup = distance_lookup if distance_lookup else {}
        self.df = df
        self.num_pois = len(venue_map)
        
        # 构建POI信息映射
        self.poi_to_category = {}
        self.poi_to_coords = {}
        self.poi_popularity = {}
        
        self._build_poi_info()
        
    def _build_poi_info(self):
        """内部辅助函数，用于从DataFrame构建POI的基础信息查找表。"""
        print("构建POI基础信息...")
        
        # POI到类别的映射
        poi_cat_df = self.df[['geo_id', 'integer_venue_category_id']].drop_duplicates()
        for _, row in poi_cat_df.iterrows():
            poi_str_id = str(row['geo_id'])
            if poi_str_id in self.venue_map:
                poi_idx = self.venue_map[poi_str_id]
                self.poi_to_category[poi_idx] = int(row['integer_venue_category_id'])
        
        # POI到坐标的映射
        poi_loc_df = self.df[['geo_id', 'latitude_norm', 'longitude_norm']].drop_duplicates()
        for _, row in poi_loc_df.iterrows():
            poi_str_id = str(row['geo_id'])
            if poi_str_id in self.venue_map:
                poi_idx = self.venue_map[poi_str_id]
                self.poi_to_coords[poi_idx] = (row['latitude_norm'], row['longitude_norm'])
        
        # POI热门度（访问次数）的映射
        poi_visit_counts = self.df['geo_id'].value_counts()
        for poi_str_id, count in poi_visit_counts.items():
            if str(poi_str_id) in self.venue_map:
                poi_idx = self.venue_map[str(poi_str_id)]
                self.poi_popularity[poi_idx] = count
                
        print(f"POI信息构建完成: {len(self.poi_to_category)}个POI-类别映射, "
              f"{len(self.poi_to_coords)}个POI-坐标映射, "
              f"{len(self.poi_popularity)}个POI-热门度映射")
    
    def build_geographical_edges(self, distance_threshold=GLOBAL_GRAPH_DISTANCE_THRESHOLD):
        """
        构建地理相邻边。
        如果两个POI之间的距离小于阈值，则在它们之间添加一条边。
        """
        print(f"构建地理相邻边 (距离阈值: {distance_threshold}km)...")
        edges = []
        for poi1_idx in self.distance_lookup:
            for poi2_idx, distance in self.distance_lookup[poi1_idx].items():
                if distance <= distance_threshold and poi1_idx != poi2_idx:
                    edges.append((poi1_idx, poi2_idx, RELATION_TYPES['GEOGRAPHICAL']))
        print(f"地理相邻边构建完成: {len(edges)}条边")
        return edges
    
    def build_same_category_edges(self, max_total_edges=1000000, max_edges_per_dense_cat=5000):
        """
        构建同类别边，并对非常密集的类别进行采样以控制边的总数。
        这可以防止图被少数几个大类别的POI完全连接所主导。
        """
        print(f"构建同类别边 (总边数上限: {max_total_edges}, 单类别边数上限: {max_edges_per_dense_cat})...")
        edges = []
        category_to_pois = defaultdict(list)
        for poi_idx, cat_id in self.poi_to_category.items():
            category_to_pois[cat_id].append(poi_idx)
        
        sorted_categories = sorted(category_to_pois.items(), key=lambda item: len(item[1]))
        total_edges_count = 0

        for cat_id, poi_list in sorted_categories:
            if len(poi_list) < 2: continue
            num_pois_in_cat = len(poi_list)
            potential_edges = num_pois_in_cat * (num_pois_in_cat - 1)

            if potential_edges <= max_edges_per_dense_cat and (total_edges_count + potential_edges) <= max_total_edges:
                for i in range(num_pois_in_cat):
                    for j in range(i + 1, num_pois_in_cat):
                        edges.append((poi_list[i], poi_list[j], RELATION_TYPES['SAME_CATEGORY']))
                        edges.append((poi_list[j], poi_list[i], RELATION_TYPES['SAME_CATEGORY']))
                total_edges_count += potential_edges
            else:
                remaining_budget = max_total_edges - total_edges_count
                edges_to_sample = min(max_edges_per_dense_cat, remaining_budget)
                if edges_to_sample <= 0: break

                edge_set = set()
                attempts, max_attempts = 0, edges_to_sample * 5
                while len(edge_set) < edges_to_sample // 2 and attempts < max_attempts:
                    try:
                        p1_idx, p2_idx = random.sample(range(num_pois_in_cat), 2)
                        edge_tuple = tuple(sorted((poi_list[p1_idx], poi_list[p2_idx])))
                        edge_set.add(edge_tuple)
                    except ValueError: break
                    attempts += 1
                
                for poi1, poi2 in edge_set:
                    edges.append((poi1, poi2, RELATION_TYPES['SAME_CATEGORY']))
                    edges.append((poi2, poi1, RELATION_TYPES['SAME_CATEGORY']))
                total_edges_count += len(edge_set) * 2

        print(f"同类别边构建完成: {len(edges)}条边")
        return edges
    
    def build_cooccurrence_edges(self, min_users=GLOBAL_GRAPH_COOCCUR_MIN_USERS):
        """
        构建共现边。
        如果两个POI被至少`min_users`个相同的用户访问过，则在它们之间添加一条边。
        """
        print(f"构建共现边 (最少共现用户数: {min_users})...")
        edges = []
        user_pois = defaultdict(set)
        for _, row in self.df.iterrows():
            poi_str_id = str(row['geo_id'])
            if poi_str_id in self.venue_map:
                user_pois[row['user_id']].add(self.venue_map[poi_str_id])
        
        poi_pair_counts = defaultdict(int)
        for poi_set in user_pois.values():
            poi_list = list(poi_set)
            for i in range(len(poi_list)):
                for j in range(i + 1, len(poi_list)):
                    poi1, poi2 = sorted((poi_list[i], poi_list[j]))
                    poi_pair_counts[(poi1, poi2)] += 1
        
        for (poi1, poi2), count in poi_pair_counts.items():
            if count >= min_users:
                edges.append((poi1, poi2, RELATION_TYPES['CO_OCCURRENCE']))
                edges.append((poi2, poi1, RELATION_TYPES['CO_OCCURRENCE']))
        
        print(f"共现边构建完成: {len(edges)}条边")
        return edges
    
    def build_popularity_edges(self, top_ratio=0.1, max_pop_edges=100000):
        """
        构建热门POI连接边。
        此方法在访问次数排名前`top_ratio`的POI之间随机创建边，
        旨在增强图中枢纽节点之间的连接。
        """
        print(f"构建热门POI连接边 (热门比例: {top_ratio}, 边数上限: {max_pop_edges})...")
        edges = []
        if not self.poi_popularity:
            print("警告: 无POI热门度信息，跳过热门边构建")
            return edges
        
        sorted_pois_with_counts = sorted(self.poi_popularity.items(), key=lambda x: x[1], reverse=True)
        num_top_pois = max(1, int(len(sorted_pois_with_counts) * top_ratio))
        top_pois_info = sorted_pois_with_counts[:num_top_pois]
        top_pois_nodes = [poi_idx for poi_idx, _ in top_pois_info]
        top_pois_weights = [count for _, count in top_pois_info]

        if len(top_pois_nodes) < 2:
            print("热门POI数量不足2个，无法构建边。")
            return []
        
        # 按热门度加权随机采样来创建边
        edge_set, count, attempts, max_attempts = set(), 0, 0, max_pop_edges * 5
        while count < max_pop_edges and attempts < max_attempts:
            try: poi1, poi2 = random.choices(top_pois_nodes, weights=top_pois_weights, k=2)
            except IndexError: break
            if poi1 == poi2: attempts += 1; continue
            
            edge_tuple = tuple(sorted((poi1, poi2)))
            if edge_tuple not in edge_set:
                edge_set.add(edge_tuple)
                edges.append((poi1, poi2, RELATION_TYPES['POPULARITY']))
                edges.append((poi2, poi1, RELATION_TYPES['POPULARITY']))
                count += 2
            attempts += 1
            
        print(f"热门POI连接边构建完成: {len(edges)}条边 (涉及{len(top_pois_nodes)}个热门POI)")
        return edges
    
    def build_global_graph(self):
        """
        构建完整的多关系全局POI图。
        
        这个方法会调用所有特定类型的边构建函数，将它们合并，去重，
        并最终返回一个PyTorch Geometric的`Data`对象。
        """
        print("\n=== 开始构建多关系全局POI图 ===")
        all_edges = []
        all_edges.extend(self.build_geographical_edges())
        if GLOBAL_GRAPH_SAME_CATEGORY_ENABLED:
            all_edges.extend(self.build_same_category_edges())
        all_edges.extend(self.build_cooccurrence_edges())
        all_edges.extend(self.build_popularity_edges())
        
        if not all_edges:
            print("警告: 未构建任何边，创建空图")
            return Data(x=self._build_node_features(), edge_index=torch.empty(2, 0, dtype=torch.long), edge_attr=torch.empty(0, dtype=torch.long), num_nodes=self.num_pois)

        # 优化：对所有收集到的边进行去重，确保(src, dst, rel)是唯一的
        unique_edges = set()
        final_edge_list, final_edge_types = [], []
        for src, dst, rel in all_edges:
            if (src, dst, rel) not in unique_edges:
                unique_edges.add((src, dst, rel))
                final_edge_list.append((src, dst))
                final_edge_types.append(rel)
        
        edge_index = torch.tensor(final_edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(final_edge_types, dtype=torch.long)
        node_features = self._build_node_features()
        
        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, num_nodes=self.num_pois)
        
        print(f"全局POI图构建完成:")
        print(f"  节点数: {graph_data.num_nodes}")
        print(f"  边数: {graph_data.num_edges}")
        for rel_name, rel_id in RELATION_TYPES.items():
            count = (edge_attr == rel_id).sum().item()
            print(f"    {rel_name}: {count}条边")
        print("=" * 50)
        
        return graph_data
    
    def _build_node_features(self):
        """
        为图中的每个POI节点构建初始特征矩阵。
        
        特征包括: [poi_id, category_id, norm_latitude, norm_longitude, log_popularity]
        这些特征将在GNN的第一层被编码为嵌入。
        """
        node_features = []
        for poi_idx in range(self.num_pois):
            cat_id = self.poi_to_category.get(poi_idx, cat_pad_idx)
            coords = self.poi_to_coords.get(poi_idx, (0.0, 0.0))
            popularity = np.log1p(self.poi_popularity.get(poi_idx, 0))
            node_features.append([poi_idx, cat_id, coords[0], coords[1], popularity])
        return torch.tensor(node_features, dtype=torch.float)

# ==============================================================================
# 1. 工具函数
# ==============================================================================
def set_seed(seed_value):
    """设置所有相关的随机种子以确保实验的可复现性。"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def generate_cyclical_encoding_table(max_val: int, embed_dim: int) -> torch.Tensor:
    """
    为周期性特征（如一天中的小时、一周中的天）生成正弦/余弦编码表。
    
    Args:
        max_val (int): 周期性特征的最大值 (例如，小时为24)。
        embed_dim (int): 编码的维度。
    
    Returns:
        torch.Tensor: 一个形状为 [max_val, embed_dim] 的编码表。
    """
    if embed_dim == 0: 
        return torch.empty(max_val, 0)
    if embed_dim % 2 != 0: 
        embed_dim +=1
    position = torch.arange(max_val, dtype=torch.float).unsqueeze(1)
    num_timescales = embed_dim // 2
    div_term = torch.exp(torch.arange(0, num_timescales, dtype=torch.float) * (-math.log(10000.0) / num_timescales))
    encoding_table = torch.zeros(max_val, embed_dim)
    encoding_table[:, 0:num_timescales] = torch.sin(position * div_term)
    if embed_dim > num_timescales:
        encoding_table[:, num_timescales:2*num_timescales] = torch.cos(position * div_term)
    return encoding_table

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    创建一个学习率调度器，该调度器在预热阶段线性增加学习率，
    然后按照余弦曲线衰减到0。
    """
    def lr_lambda(current_step):
        if num_training_steps <= 0: return 0.0
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_time_segment_id(timestamp: pd.Timestamp) -> int:
    """根据预定义的规则，将一个时间戳映射到一个离散的时间段ID。"""
    if pd.isna(timestamp): return TIME_SEGMENT_PAD_IDX
    weekday, hour = timestamp.weekday(), timestamp.hour
    for seg_name, rule_func in TIME_SEGMENT_RULES:
        if rule_func(weekday, hour): return TIME_SEGMENT_CATEGORIES[seg_name]
    return TIME_SEGMENT_CATEGORIES.get("LATE_NIGHT_DEEP", NUM_TIME_SEGMENTS - 1)

def bin_series_to_int(series, bins, labels, pad_value):
    """将一个Pandas Series中的连续值根据给定的分箱规则离散化为整数标签。"""
    if not isinstance(series, pd.Series): series = pd.Series(series)
    binned_series = pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)
    return [int(x) if pd.notna(x) else pad_value for x in binned_series]

def bin_pairwise_time_diffs_torch(pairwise_diff_minutes_tensor, bins_list):
    """
    使用PyTorch高效地对成对时间差张量进行分箱。
    这比Pandas的pd.cut快得多，适用于在模型内部处理。
    """
    boundaries_tensor = torch.tensor(bins_list[1:-1], dtype=pairwise_diff_minutes_tensor.dtype, device=pairwise_diff_minutes_tensor.device)
    return torch.bucketize(pairwise_diff_minutes_tensor, boundaries_tensor, right=False)

def load_precomputed_distances(distance_file_path):
    """
    从CSV文件加载预先计算好的POI间距离，并构建一个高效的查找字典。
    """
    global venue_map
    if not venue_map:
        print("错误: venue_map 未初始化，无法加载距离数据。")
        return None
    print(f"加载预计算的距离数据从: {distance_file_path}")
    try: dist_df = pd.read_csv(distance_file_path)
    except FileNotFoundError: print(f"错误: 距离文件未找到于 {distance_file_path}"); return None
    
    dist_lookup = defaultdict(dict)
    max_venue_idx = max(venue_map.values()) if venue_map else -1
    
    for _, row in tqdm(dist_df.iterrows(), total=len(dist_df), desc="构建距离查找表"):
        try:
            v1_idx, v2_idx, dist = int(row['venue1']), int(row['venue2']), float(row['distance'])
            if 0 <= v1_idx <= max_venue_idx and 0 <= v2_idx <= max_venue_idx:
                dist_lookup[v1_idx][v2_idx] = dist
                dist_lookup[v2_idx][v1_idx] = dist
        except (ValueError, KeyError):
            continue
    print(f"距离查找表构建完成，包含 {len(dist_lookup)} 个POI的距离信息。")
    return dist_lookup

def info_nce_loss(anchor_embeds, positive_embeds, negative_embeds, temperature=0.1):
    """
    计算InfoNCE对比损失。
    目标是让锚点（anchor）的嵌入与正样本（positive）的嵌入更接近，
    同时与所有负样本（negative）的嵌入更疏远。
    
    Args:
        anchor_embeds (Tensor): 锚点样本的嵌入 [B, D]。
        positive_embeds (Tensor): 正样本的嵌入 [B, D]。
        negative_embeds (Tensor): 负样本的嵌入 [B, N, D]。
        temperature (float): 温度系数，用于缩放logits。
    
    Returns:
        Tensor: 计算出的InfoNCE损失。
    """
    positive_embeds = positive_embeds.unsqueeze(1) # [B, 1, D]
    all_candidates = torch.cat([positive_embeds, negative_embeds], dim=1) # [B, 1+N, D]
    logits = torch.bmm(anchor_embeds.unsqueeze(1), all_candidates.transpose(1, 2)).squeeze(1) # [B, 1+N]
    logits = logits / temperature
    labels = torch.zeros(anchor_embeds.size(0), dtype=torch.long, device=anchor_embeds.device)
    return F.cross_entropy(logits, labels)

# ==============================================================================
# 2. 数据加载与预处理函数
# ==============================================================================
def load_and_preprocess_data(data_path: str):
    """
    加载原始CSV数据并进行全面的预处理。
    
    步骤包括：
    1. 加载数据并解析时间戳。
    2. 按用户和时间排序。
    3. 计算POI热门度。
    4. 归一化经纬度坐标。
    5. 为POI、类别和用户创建唯一的整数ID映射。
    6. 设置全局变量，如实体数量和填充索引。
    
    Returns:
        pd.DataFrame: 经过预处理的DataFrame。
    """
    global num_categories, cat_pad_idx, num_categories_with_pad, \
           num_venues, venue_pad_idx, num_venues_with_pad, \
           user_map, venue_map, category_map
    print("加载并预处理数据...")
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df['hour']=df['time'].dt.hour
    df['weekday']=df['time'].dt.weekday
    df = df.sort_values(by=['user_id', 'time'])
    
    # 计算POI热门度（对数变换以平滑分布）
    poi_visit_counts = df['geo_id'].value_counts()
    df['poi_popularity_log'] = np.log1p(df['geo_id'].map(poi_visit_counts).fillna(0))
    
    # 归一化经纬度到[0, 1]范围
    lat_min,lat_max=df['latitude'].min(),df['latitude'].max() 
    lon_min,lon_max=df['longitude'].min(),df['longitude'].max()
    df['latitude_norm']=(df['latitude']-lat_min)/(lat_max-lat_min) if lat_max>lat_min else 0.0
    df['longitude_norm']=(df['longitude']-lon_min)/(lon_max-lon_min) if lon_max>lon_min else 0.0
    
    # 创建类别ID映射
    df['venue_category_id']=df['venue_category_id'].fillna('UNK_CAT').astype(str)
    unique_categories_val=df['venue_category_id'].unique()
    category_map={cid:i for i,cid in enumerate(unique_categories_val)}
    num_categories=len(category_map) 
    cat_pad_idx=num_categories
    num_categories_with_pad=num_categories+1
    df['integer_venue_category_id']=df['venue_category_id'].map(category_map)
    
    # 创建POI ID映射
    df['geo_id'] = df['geo_id'].astype(str)
    unique_venues_list=df['geo_id'].unique()
    venue_map={vid:i for i,vid in enumerate(unique_venues_list)}
    num_venues=len(venue_map)
    venue_pad_idx=num_venues
    num_venues_with_pad=num_venues+1

    # 创建用户ID映射
    all_user_ids_in_df = df['user_id'].unique()
    user_map = {uid: i for i, uid in enumerate(all_user_ids_in_df)}
    print(f"数据预处理完成。用户数:{len(user_map)}, 类别数:{num_categories}, POI数:{num_venues}")
    return df

def split_users(df: pd.DataFrame, val_ratio: float, test_ratio: float):
    """
    按用户ID将数据集划分为训练集、验证集和测试集。
    这确保了来自同一用户的轨迹不会同时出现在不同的数据集中。
    
    Returns:
        tuple: 包含训练、验证、测试用户ID集合的元组。
    """
    all_user_ids=df['user_id'].unique()
    np.random.shuffle(all_user_ids)
    n_users=len(all_user_ids)
    n_val=int(n_users*val_ratio)
    n_test=int(n_users*test_ratio)
    val_ids=set(all_user_ids[:n_val])
    test_ids=set(all_user_ids[n_val:n_val+n_test])
    train_ids=set(all_user_ids[n_val+n_test:])
    if not train_ids and n_users > 0: train_ids = val_ids if not val_ids else set(all_user_ids)
    print(f"用户划分完成。训练集:{len(train_ids)}, 验证集:{len(val_ids)}, 测试集:{len(test_ids)}")
    return train_ids, val_ids, test_ids

# ==============================================================================
# 3. Dataset 和 DataLoader 定义
# ==============================================================================
class TrajectoryDataset(Dataset):
    """
    用于主轨迹预测任务的PyTorch数据集。
    
    它将每个用户的完整轨迹分解成多个`(输入序列, 目标)`对。
    对于用户轨迹中的每个点`t`，它会创建一个样本，其中输入是`t`之前的轨迹（最多`max_len`），
    目标是点`t`的POI类别。
    
    在训练模式下，它还可以对输入序列应用随机masking作为一种数据增强。
    """
    def __init__(self, data_grouped, venue_map_g, max_len, venue_pad_idx_g, cat_pad_idx_g, num_categories_g,
                 masking_ratio=0.0, is_train=False):
        self.data = []
        self.max_len = max_len
        self.venue_map = venue_map_g
        self.venue_pad_idx = venue_pad_idx_g
        self.cat_pad_idx = cat_pad_idx_g
        self.num_categories = num_categories_g
        
        self.masking_ratio = masking_ratio
        self.is_train = is_train

        print(f"为 {len(data_grouped)} 用户处理轨迹 (主任务)...")
        if self.is_train and self.masking_ratio > 0:
            print(f"  启用随机Masking，比例: {self.masking_ratio:.2f}")

        for user_id, group in tqdm(data_grouped, desc="创建主任务序列"):
            group = group.sort_values('time')
            if len(group) < 2: continue # 至少需要一个历史点和一个目标点
                
            # --- 提取一个用户的完整轨迹特征 ---
            venues_ids = [self.venue_map.get(str(v), self.venue_pad_idx) for v in group['geo_id'].tolist()]
            hours = group['hour'].tolist()
            time_segment_type_ids = [get_time_segment_id(ts) for ts in group['time']]
            cats = [int(c) if pd.notna(c) and 0<=int(c)<self.num_categories else self.cat_pad_idx for c in group['integer_venue_category_id'].tolist()]
            norm_lats = group['latitude_norm'].tolist()
            norm_lons = group['longitude_norm'].tolist()
            popularities = group['poi_popularity_log'].tolist()
            raw_timestamps_utc = group['time'].apply(lambda x:x.timestamp() if pd.notna(x) else 0.0).tolist()
            
            # --- 滑动窗口生成样本 ---
            for i in range(1, len(venues_ids)):
                input_end_idx = i
                start_idx = max(0, input_end_idx - self.max_len)
                
                 # --- 提取原始序列 (无mask)，用于辅助任务目标 ---
                seq_v_orig = venues_ids[start_idx:input_end_idx]
                seq_h_orig = hours[start_idx:input_end_idx]
                seq_ts_type_orig = time_segment_type_ids[start_idx:input_end_idx]
                seq_c_orig = cats[start_idx:input_end_idx]
                seq_nl_orig = norm_lats[start_idx:input_end_idx]
                seq_nlo_orig = norm_lons[start_idx:input_end_idx]
                seq_pop_orig = popularities[start_idx:input_end_idx]
                seq_raw_ts_orig = raw_timestamps_utc[start_idx:input_end_idx]
                
                target_cat = cats[i] # 主任务目标
                
                # --- 辅助任务的目标是重建原始序列 ---
                aux_cats_target, aux_venues_target = copy.deepcopy(seq_c_orig), copy.deepcopy(seq_v_orig)
                
                if not seq_v_orig or target_cat == self.cat_pad_idx: continue

                # 应用随机Masking作为数据增强
                seq_v = list(seq_v_orig) # 创建一个可修改的副本
                seq_c = list(seq_c_orig)
                seq_h = list(seq_h_orig)
                seq_ts_type = list(seq_ts_type_orig)
                seq_nl = list(seq_nl_orig)
                seq_nlo = list(seq_nlo_orig)
                seq_pop = list(seq_pop_orig)
                seq_raw_ts = list(seq_raw_ts_orig)
                
                # 只在训练时且masking_ratio > 0时应用
                if self.is_train and self.masking_ratio > 0:
                    seq_len = len(seq_v)
                    num_to_mask = int(seq_len * self.masking_ratio)
                    
                    if num_to_mask > 0:
                        # 随机选择要mask掉的位置索引
                        mask_indices = random.sample(range(seq_len), k=num_to_mask)
                        
                        for mask_idx in mask_indices:
                            # 将这个位置的所有特征都替换为它们的"padding"或"neutral"值
                            seq_v[mask_idx] = self.venue_pad_idx
                            seq_c[mask_idx] = self.cat_pad_idx
                            seq_h[mask_idx] = 0
                            seq_ts_type[mask_idx] = TIME_SEGMENT_PAD_IDX
                            seq_nl[mask_idx] = 0.0
                            seq_nlo[mask_idx] = 0.0
                            seq_pop[mask_idx] = 0.0
                            seq_raw_ts[mask_idx] = 0.0

                self.data.append({
                    'user_id': user_id,
                    'venues': seq_v, # 使用可能被mask过的序列
                    'hours': seq_h,
                    'time_segment_types': seq_ts_type,
                    'cats': seq_c,
                    'lats': seq_nl,
                    'lons': seq_nlo,
                    'popularities': seq_pop,
                    'raw_timestamps': seq_raw_ts,
                    'target': target_cat,
                    'aux_cats': aux_cats_target, # 辅助任务目标仍然是原始序列
                    'aux_venues': aux_venues_target
                })

        print(f"创建了 {len(self.data)} 个主任务序列样本。")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]

def main_task_collate_fn(batch, venue_pad_idx_g, cat_pad_idx_g, time_segment_pad_idx_g):
    """
    用于主任务数据集的collate函数。
    
    它将一批样本（每个都是一个字典）处理成一个批次化的字典，其中包含：
    1. 填充到批次中最大长度的序列张量。
    2. 一个布尔型的`padding_mask`张量，标记出填充的位置。
    3. 其他元数据，如用户ID和目标。
    """
    keys = batch[0].keys() if batch else []
    batch_data = {k:[] for k in keys}
    seq_lens = []
    valid_batch = [item for item in batch if item and 'venues' in item and item['venues']]
    if not valid_batch: return None
    batch = valid_batch
    max_len_in_batch = 0
    if batch: max_len_in_batch = max(len(item['venues']) for item in batch if 'venues' in item and item['venues'])
    if max_len_in_batch == 0: return None
    
    for item in batch:
        seq_len = len(item['venues'])
        seq_lens.append(seq_len)
        pad_len = max_len_in_batch - seq_len
        
        # 填充序列特征
        for k in ['venues','cats','aux_cats','aux_venues']:
            pad_val = venue_pad_idx_g if 'venue' in k else cat_pad_idx_g
            batch_data[k].append(torch.tensor(item[k] + [pad_val]*pad_len, dtype=torch.long))
        
        batch_data['hours'].append(torch.tensor(item['hours'] + [0]*pad_len, dtype=torch.long))
        
        batch_data['time_segment_types'].append(torch.tensor(item['time_segment_types'] + [time_segment_pad_idx_g]*pad_len, dtype=torch.long))
        
        for k in ['lats','lons','popularities','raw_timestamps']:
            dtype = torch.double if k=='raw_timestamps' else torch.float
            batch_data[k].append(torch.tensor(item[k] + [0.0]*pad_len, dtype=dtype))
        
        for k in ['user_id','target']: batch_data[k].append(item[k])
    
    # 将列表中的张量堆叠成一个批次
    stacked_data={}
    tensor_keys=['venues','hours','time_segment_types','cats','lats','lons','popularities','raw_timestamps','aux_cats','aux_venues']
    
    for k in tensor_keys:
        if k in batch_data and batch_data[k]: 
            stacked_data[k] = torch.stack(batch_data[k])
    
    if 'user_id' in batch_data: stacked_data['user_ids'] = batch_data['user_id']
    if 'target' in batch_data: stacked_data['target'] = torch.tensor(batch_data['target'], dtype=torch.long)
    stacked_data['seq_lens']=torch.tensor(seq_lens,dtype=torch.long)
    
    # 创建padding mask
    if 'venues' in stacked_data and stacked_data['venues'].numel() > 0: 
        stacked_data['padding_mask']=(stacked_data['venues']==venue_pad_idx_g)
    else: 
        bs = len(batch_data.get('user_ids', batch))
        seq_l = max_len_in_batch if max_len_in_batch > 0 else 1
        stacked_data['padding_mask']=torch.ones(bs, seq_l, dtype=torch.bool)
    
    return stacked_data

class UserFullHistoryDataset(Dataset):
    """
    用于用户表示学习的PyTorch数据集。
    
    对于每个用户，此数据集提取其完整的（或最近的`max_len_traj`）历史轨迹。
    它将这个历史表示为两种形式：
    1. 一个序列，包含POI ID、时间、类别等特征。
    2. 一个局部的、动态的图，其中节点是轨迹中访问过的独特POI，
       边代表连续的签到。边的属性包括时间差和地理距离。
       
    在训练时，可以对图的边进行随机丢弃（edge dropout）作为数据增强。
    """
    def __init__(self, df_user_histories_grouped: pd.core.groupby.generic.DataFrameGroupBy,
                 venue_map_g: dict, category_map_g: dict,
                 max_len_traj: int,
                 venue_pad_idx_g: int, cat_pad_idx_g: int,
                 edge_time_bins_list: list, edge_dist_bins_list: list,
                 distance_lookup_g: dict,
                 is_train: bool = False,
                 edge_dropout_rate: float = 0.0
                ):
        self.user_data_list = []
        self.max_len_traj = max_len_traj
        self.venue_map = venue_map_g
        self.category_map = category_map_g
        self.venue_pad_idx = venue_pad_idx_g
        self.cat_pad_idx = cat_pad_idx_g

        self.edge_time_bins_list = edge_time_bins_list
        self.edge_dist_bins_list = edge_dist_bins_list
        self.edge_time_labels = list(range(len(edge_time_bins_list)-1)) if edge_time_bins_list else []
        self.edge_dist_labels = list(range(len(edge_dist_bins_list)-1)) if edge_dist_bins_list else []
        self.distance_lookup = distance_lookup_g if distance_lookup_g is not None else {}

        self.is_train = is_train
        self.edge_dropout_rate = edge_dropout_rate

        print(f"为用户画像 (GATv2) Dataset 处理 {len(df_user_histories_grouped)} 个用户...")
        for user_id, group in tqdm(df_user_histories_grouped, desc="用户画像数据构建(GATv2)"):
            group = group.sort_values('time').tail(self.max_len_traj)
            if len(group) < 2:
                continue

            # --- 1. 提取序列特征 ---
            seq_venues_ids = [self.venue_map.get(str(v), self.venue_pad_idx) for v in group['geo_id'].tolist()]
            hours_seq = group['hour'].tolist()
            time_segment_types_seq = [get_time_segment_id(ts) for ts in group['time']]
            cats_seq = [int(c) if pd.notna(c) and 0 <= c < len(self.category_map) else self.cat_pad_idx
                        for c in group['integer_venue_category_id'].tolist()]
            lats_seq = group['latitude_norm'].tolist()
            lons_seq = group['longitude_norm'].tolist()
            popularities_seq = group['poi_popularity_log'].tolist()
            raw_timestamps_seq = group['time'].apply(lambda x: x.timestamp() if pd.notna(x) else 0.0).tolist()
            raw_times_pd_seq = group['time'].tolist()

            # --- 2. 构建GAT图结构 ---
            unique_pois_df_in_group = group.drop_duplicates(subset=['geo_id'])
            
            node_global_venue_ids = []
            node_global_cat_ids = []
            node_norm_lats = []
            node_norm_lons = []
            node_popularities = []
            
            map_global_venue_idx_to_local = {}
            
            current_local_node_idx = 0
            for _, poi_row in unique_pois_df_in_group.iterrows():
                global_venue_idx = self.venue_map.get(str(poi_row['geo_id']), self.venue_pad_idx)
                if global_venue_idx == self.venue_pad_idx: continue 

                if global_venue_idx not in map_global_venue_idx_to_local:
                    map_global_venue_idx_to_local[global_venue_idx] = current_local_node_idx
                    node_global_venue_ids.append(global_venue_idx)
                    
                    cat_val_int = poi_row['integer_venue_category_id']
                    node_global_cat_ids.append(int(cat_val_int) if pd.notna(cat_val_int) and 0 <= cat_val_int < len(self.category_map) else self.cat_pad_idx)
                    
                    node_norm_lats.append(poi_row['latitude_norm'])
                    node_norm_lons.append(poi_row['longitude_norm'])
                    node_popularities.append(poi_row.get('poi_popularity_log', 0.0))
                    current_local_node_idx += 1
            
            if not node_global_venue_ids or len(node_global_venue_ids) < 1:
                continue

            # --- 3. 构建边和边属性 ---
            edge_list_src_local_idx, edge_list_dst_local_idx = [], []
            edge_time_attr_list, edge_dist_attr_list = [], []

            if len(group) >= 2:
                # 遍历连续的签到对来创建边
                for i in range(len(group) - 1):
                    row_u, row_v = group.iloc[i], group.iloc[i+1]
                    u_gid = self.venue_map.get(str(row_u['geo_id']), self.venue_pad_idx)
                    v_gid = self.venue_map.get(str(row_v['geo_id']), self.venue_pad_idx)

                    if u_gid != self.venue_pad_idx and v_gid != self.venue_pad_idx and u_gid != v_gid:
                        if u_gid in map_global_venue_idx_to_local and v_gid in map_global_venue_idx_to_local:
                            u_local = map_global_venue_idx_to_local[u_gid]
                            v_local = map_global_venue_idx_to_local[v_gid]
                            edge_list_src_local_idx.append(u_local)
                            edge_list_dst_local_idx.append(v_local)

                            # 计算并分箱时间间隔属性
                            time_diff_min = (raw_times_pd_seq[i+1] - raw_times_pd_seq[i]).total_seconds() / 60.0
                            time_bin = bin_series_to_int(pd.Series([time_diff_min]), self.edge_time_bins_list, self.edge_time_labels, EDGE_TIME_BIN_PAD_IDX_GAT)[0]
                            edge_time_attr_list.append(time_bin)

                            # 计算并分箱距离属性
                            dist_km = -1.0 
                            if self.distance_lookup: 
                                if u_gid in self.distance_lookup and v_gid in self.distance_lookup[u_gid]:
                                    dist_km = self.distance_lookup[u_gid][v_gid]
                                elif v_gid in self.distance_lookup and u_gid in self.distance_lookup[v_gid]: 
                                    dist_km = self.distance_lookup[v_gid][u_gid]
                            
                            dist_bin = bin_series_to_int(pd.Series([dist_km if dist_km >=0 else -1.0]), self.edge_dist_bins_list, self.edge_dist_labels, EDGE_DIST_BIN_PAD_IDX_GAT)[0]
                            edge_dist_attr_list.append(dist_bin)
            
            # --- 4. 边丢弃（仅在训练时） ---
            if self.is_train and self.edge_dropout_rate > 0 and len(edge_list_src_local_idx) > 0:
                num_edges = len(edge_list_src_local_idx)
                perm = np.random.permutation(num_edges)
                num_edges_to_keep = int(num_edges * (1.0 - self.edge_dropout_rate))
                indices_to_keep = perm[:num_edges_to_keep]
                
                edge_list_src_local_idx = [edge_list_src_local_idx[i] for i in indices_to_keep]
                edge_list_dst_local_idx = [edge_list_dst_local_idx[i] for i in indices_to_keep]
                edge_time_attr_list = [edge_time_attr_list[i] for i in indices_to_keep]
                edge_dist_attr_list = [edge_dist_attr_list[i] for i in indices_to_keep]
            
            # --- 5. 创建GAT数据对象 ---
            gat_data_obj = Data(
                x_node_venue_ids=torch.tensor(node_global_venue_ids, dtype=torch.long),
                x_node_cat_ids=torch.tensor(node_global_cat_ids, dtype=torch.long),
                x_node_locs=torch.tensor(list(zip(node_norm_lats, node_norm_lons)), dtype=torch.float),
                x_node_popularity=torch.tensor(node_popularities, dtype=torch.float).unsqueeze(1),
                edge_index=torch.tensor([edge_list_src_local_idx, edge_list_dst_local_idx], dtype=torch.long) if edge_list_src_local_idx else torch.empty(2,0,dtype=torch.long),
                edge_attr=torch.tensor(list(zip(edge_time_attr_list, edge_dist_attr_list)), dtype=torch.long) if edge_list_src_local_idx else torch.empty(0,2,dtype=torch.long)
            )

            self.user_data_list.append({
                'user_id': user_id,
                # 序列部分
                'venues_seq': seq_venues_ids,
                'hours_seq': hours_seq,
                'time_segment_types_seq': time_segment_types_seq, 
                'cats_seq': cats_seq,
                'lats_seq': lats_seq, 
                'lons_seq': lons_seq, 
                'popularities_seq': popularities_seq,
                'raw_timestamps_seq': raw_timestamps_seq,
                # 图部分
                'gat_data': gat_data_obj
            })

    def __len__(self): return len(self.user_data_list)

    def __getitem__(self, idx): return self.user_data_list[idx]

def user_full_history_gat_collate_fn(batch, venue_pad_idx_g, cat_pad_idx_g, time_segment_pad_idx_g):
    """
    用于UserFullHistoryDataset的collate函数。
    
    它能同时处理序列数据和图数据：
    - 对序列数据进行填充，类似于`main_task_collate_fn`。
    - 使用`torch_geometric.data.Batch.from_data_list`将批次中的所有局部图
      合并成一个大的、不连通的图，这是PyG处理批次图的常用方法。
    """
    seq_keys_to_process = [
        'venues_seq', 'hours_seq', 'time_segment_types_seq',
        'cats_seq', 'lats_seq', 'lons_seq', 'popularities_seq', 'raw_timestamps_seq'
    ]
    batch_seq_data = {k: [] for k in seq_keys_to_process}
    batch_seq_data['user_ids'] = []
    seq_lens_list = []

    valid_batch_items = [item for item in batch if item is not None and isinstance(item, dict)]
    if not valid_batch_items:
        return None
    batch = valid_batch_items

    max_seq_len_in_batch = 0
    if any(item.get('venues_seq') for item in batch): 
        max_seq_len_in_batch = max(len(item.get('venues_seq',[])) for item in batch)
    
    gat_data_list_for_pyg_batch = []

    for item in batch:
        # --- 填充序列数据 ---
        seq_len = len(item.get('venues_seq', []))
        seq_lens_list.append(seq_len)
        pad_len = max_seq_len_in_batch - seq_len
        
        batch_seq_data['user_ids'].append(item.get('user_id', -1))

        for k_seq in ['venues_seq', 'cats_seq']:
            item_k_val = item.get(k_seq, [])
            pad_val = venue_pad_idx_g if 'venue' in k_seq else cat_pad_idx_g
            batch_seq_data[k_seq].append(torch.tensor(item_k_val + [pad_val] * pad_len, dtype=torch.long))
        
        item_h_val = item.get('hours_seq', [])
        batch_seq_data['hours_seq'].append(torch.tensor(item_h_val + [0] * pad_len, dtype=torch.long))
        
        item_ts_val = item.get('time_segment_types_seq', [])
        batch_seq_data['time_segment_types_seq'].append(torch.tensor(item_ts_val + [time_segment_pad_idx_g] * pad_len, dtype=torch.long))
        
        for k_seq in ['lats_seq', 'lons_seq', 'popularities_seq', 'raw_timestamps_seq']:
            item_k_val = item.get(k_seq, [])
            dtype_val = torch.double if 'timestamp' in k_seq else torch.float
            batch_seq_data[k_seq].append(torch.tensor(item_k_val + [0.0] * pad_len, dtype=dtype_val))

        # --- 处理GAT数据 ---
        current_gat_data = item.get('gat_data')
        
        default_empty_gat_data = Data(
            x_node_venue_ids=torch.empty(0, dtype=torch.long),
            x_node_cat_ids=torch.empty(0, dtype=torch.long),
            x_node_locs=torch.empty(0, 2, dtype=torch.float),
            x_node_popularity=torch.empty(0, 1, dtype=torch.float),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, 2, dtype=torch.long)
        )

        if current_gat_data is None or not hasattr(current_gat_data, 'x_node_venue_ids') or current_gat_data.x_node_venue_ids.numel() == 0:
            data_for_pyg_batch = default_empty_gat_data.clone()
        else:
            data_for_pyg_batch = current_gat_data.clone()
            
            for attr_name, default_tensor in default_empty_gat_data.to_dict().items():
                if not hasattr(data_for_pyg_batch, attr_name) or getattr(data_for_pyg_batch, attr_name) is None:
                    setattr(data_for_pyg_batch, attr_name, default_tensor.clone())
            
            if hasattr(data_for_pyg_batch, 'edge_index') and data_for_pyg_batch.edge_index.size(1) > 0 and \
               (not hasattr(data_for_pyg_batch, 'edge_attr') or data_for_pyg_batch.edge_attr.numel() == 0):
                num_edges = data_for_pyg_batch.edge_index.size(1)
                setattr(data_for_pyg_batch, 'edge_attr', 
                        torch.full((num_edges, 2), 
                                   fill_value=EDGE_TIME_BIN_PAD_IDX_GAT,
                                   dtype=torch.long))

        gat_data_list_for_pyg_batch.append(data_for_pyg_batch)

    # --- 堆叠数据 ---
    stacked_data = {}
    for k_seq in seq_keys_to_process:
        if k_seq in batch_seq_data and batch_seq_data[k_seq]:
            try:
                stacked_data[k_seq] = torch.stack(batch_seq_data[k_seq])
            except RuntimeError:
                return None 
    
    stacked_data['user_ids'] = batch_seq_data['user_ids']
    stacked_data['seq_lens'] = torch.tensor(seq_lens_list, dtype=torch.long)
    
    if 'venues_seq' in stacked_data and stacked_data['venues_seq'].numel() > 0 :
        stacked_data['padding_mask_seq'] = (stacked_data['venues_seq'] == venue_pad_idx_g)
    else:
        dummy_bs = len(batch_seq_data['user_ids']) if batch_seq_data.get('user_ids') else (len(batch) if batch else 1)
        dummy_seq_len = max_seq_len_in_batch if max_seq_len_in_batch > 0 else 1
        stacked_data['padding_mask_seq'] = torch.ones(dummy_bs, dummy_seq_len, dtype=torch.bool)

    if gat_data_list_for_pyg_batch:
        try:
            stacked_data['gat_batch'] = Batch.from_data_list(gat_data_list_for_pyg_batch)
        except Exception as e:
            print(f"Error in Batch.from_data_list: {e}")
            stacked_data['gat_batch'] = Batch()
    else:
        stacked_data['gat_batch'] = Batch()
        
    return stacked_data

# ==============================================================================
# 4. 模型定义
# ==============================================================================
class PositionalEncoding(nn.Module):
    """固定的正弦/余弦位置编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1); num_timescales = d_model // 2
        div_term = torch.exp(torch.arange(0, num_timescales, dtype=torch.float) * (-math.log(10000.0) / num_timescales))
        pe = torch.zeros(1, max_len, d_model)
        pe[0,:, 0:2*num_timescales:2] = torch.sin(position*div_term); pe[0,:, 1:2*num_timescales:2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe) # 注册为buffer，它会随模型移动但不是参数
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.dropout(x + self.pe[:, :x.size(1), :])

class AttentionPooling(nn.Module):
    """
    注意力池化层。
    它学习一个权重，对序列中每个时间步的输出进行加权平均，
    而不是简单的平均池化或取最后一个输出。
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention_net = nn.Sequential(nn.Linear(input_dim, input_dim//2), nn.Tanh(), nn.Linear(input_dim//2, 1))
    def forward(self, x, mask):
        attn_logits = self.attention_net(x).squeeze(-1)
        attn_logits.masked_fill_(mask, -float('inf'))
        attn_weights = F.softmax(attn_logits, dim=1)
        return torch.sum(x * (attn_weights.unsqueeze(-1) + 1e-9), dim=1)

class TimeAwareScaledDotProductAttention(nn.Module):
    """
    带时间偏差的缩放点积注意力。
    这是标准注意力机制的一个变体，它在计算注意力分数后，
    直接加上一个可学习的时间偏差项。
    """
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, time_bias=None, key_padding_mask=None, attn_mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if time_bias is not None: scores = scores + time_bias
        if key_padding_mask is not None: scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        if attn_mask is not None:
            if attn_mask.dim()==2: attn_mask_expanded=attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim()==3: attn_mask_expanded=attn_mask.unsqueeze(1)
            else: attn_mask_expanded=attn_mask
            if attn_mask_expanded.dtype==torch.bool: scores=scores.masked_fill(attn_mask_expanded,float('-inf'))
            else: scores=scores+attn_mask_expanded
        p_attn = F.softmax(scores, dim=-1); p_attn = torch.nan_to_num(p_attn, nan=0.0)
        p_attn = self.dropout(p_attn); return torch.matmul(p_attn, v), p_attn

class TimeAwareMultiheadAttention_ContextGated(nn.Module):
    """
    带上下文门控的时间感知多头注意力。
    
    这是一个复杂的自定义注意力模块，它不仅加入了时间偏差，还引入了一个
    “上下文门控”机制。这个门控基于查询（Q）和键（K）本身来动态地
    调整时间偏差的强度。这使得模型可以根据当前上下文决定时间信息的重要性。
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, num_pairwise_time_bins=None, bias_in_linear=True,
                 gate_activation='sigmoid', context_gate_input_type='qk'):
        super().__init__()
        if num_pairwise_time_bins is None: raise ValueError("必须提供 num_pairwise_time_bins 参数。")
        self.embed_dim = embed_dim; self.num_heads = num_heads; self.head_dim = embed_dim // num_heads
        if self.head_dim == 0 and embed_dim > 0 and num_heads > 0 : self.head_dim = 1
        if self.num_heads == 0 and embed_dim > 0: self.num_heads = 1; self.head_dim = embed_dim
        if self.head_dim * self.num_heads != self.embed_dim and embed_dim > 0 :
            print(f"警告: embed_dim ({embed_dim}) 不能被 num_heads ({num_heads})完美整除。")
            self.embed_dim = self.head_dim * self.num_heads

        self.q_proj=nn.Linear(embed_dim,embed_dim,bias=bias_in_linear)
        self.k_proj=nn.Linear(embed_dim,embed_dim,bias=bias_in_linear)
        self.v_proj=nn.Linear(embed_dim,embed_dim,bias=bias_in_linear)
        self.out_proj=nn.Linear(embed_dim,embed_dim,bias=bias_in_linear)
        self.scaled_dot_product_attention = TimeAwareScaledDotProductAttention(dropout=dropout)
        # 时间偏差嵌入层：将离散的时间差bin映射为向量
        self.time_diff_bias_embedding = nn.Embedding(num_pairwise_time_bins, num_heads)
        nn.init.zeros_(self.time_diff_bias_embedding.weight); self.time_bias_layernorm = nn.LayerNorm(num_heads)
        self.gate_activation_type=gate_activation
        self.context_gate_input_type=context_gate_input_type
        gate_input_feature_dim = self.head_dim*2
        if self.context_gate_input_type=='qkt': gate_input_feature_dim += num_heads
        
        self.contextual_gate_fc_sigmoid = nn.Linear(max(1,gate_input_feature_dim), num_heads)
        nn.init.xavier_uniform_(self.contextual_gate_fc_sigmoid.weight,gain=nn.init.calculate_gain('sigmoid'))
        if self.contextual_gate_fc_sigmoid.bias is not None: nn.init.zeros_(self.contextual_gate_fc_sigmoid.bias)
        if self.gate_activation_type=='tanh_sigmoid':
            self.contextual_gate_fc_tanh = nn.Linear(max(1,gate_input_feature_dim), num_heads)
            nn.init.xavier_uniform_(self.contextual_gate_fc_tanh.weight,gain=nn.init.calculate_gain('tanh'))
            if self.contextual_gate_fc_tanh.bias is not None: nn.init.zeros_(self.contextual_gate_fc_tanh.bias)

    def forward(self, query, key, value, binned_pairwise_time_diffs, key_padding_mask=None, attn_mask=None):
        # 标准的多头注意力Q, K, V投影
        B,L_q,E=query.shape; _,L_k,_=key.shape
        if E == 0: return value, None 
        if self.num_heads == 0: return self.out_proj(value), None

        q_proj=self.q_proj(query); k_proj=self.k_proj(key); v_proj=self.v_proj(value)
        q=q_proj.view(B,L_q,self.num_heads,self.head_dim).transpose(1,2)
        k=k_proj.view(B,L_k,self.num_heads,self.head_dim).transpose(1,2)
        v=v_proj.view(B,L_k,self.num_heads,self.head_dim).transpose(1,2)

        # 获取原始时间偏差
        time_bias_raw=self.time_diff_bias_embedding(binned_pairwise_time_diffs)
        time_bias_normalized=self.time_bias_layernorm(time_bias_raw)

        # 计算门控值
        q_exp=q.permute(0,2,1,3).unsqueeze(2).expand(-1,-1,L_k,-1,-1)
        k_exp=k.permute(0,2,1,3).unsqueeze(1).expand(-1,L_q,-1,-1,-1)
        qk_concat=torch.cat([q_exp,k_exp],dim=-1) # 将Q和K拼接作为门控网络的输入
        gate_fc_in_feat=qk_concat
        if self.context_gate_input_type=='qkt':
            tb_raw_exp=time_bias_raw.unsqueeze(3).expand(-1,-1,-1,self.num_heads,-1)
            gate_fc_in_feat=torch.cat([gate_fc_in_feat,tb_raw_exp],dim=-1)
        mean_feat_gate=gate_fc_in_feat.mean(dim=3) # 对head_dim维度取平均
        logits_sig=self.contextual_gate_fc_sigmoid(mean_feat_gate)
        gate_vals=None
        if self.gate_activation_type=='sigmoid': gate_vals=torch.sigmoid(logits_sig)
        elif self.gate_activation_type=='tanh_sigmoid':
            if not hasattr(self,'contextual_gate_fc_tanh'): raise AttributeError("tanh_sigmoid selected, but fc_tanh missing.")
            logits_tanh=self.contextual_gate_fc_tanh(mean_feat_gate)
            gate_vals=torch.sigmoid(logits_sig)*torch.tanh(logits_tanh)
        elif self.gate_activation_type=='identity': gate_vals=logits_sig
        else: gate_vals=torch.sigmoid(logits_sig)

        # 应用门控
        time_bias_gated=time_bias_normalized*gate_vals
        time_bias=time_bias_gated.permute(0,3,1,2)

        # 调用底层注意力
        attn_out,attn_w=self.scaled_dot_product_attention(q,k,v,time_bias,key_padding_mask,attn_mask)
        attn_out=attn_out.transpose(1,2).contiguous().view(B,L_q,E)
        return self.out_proj(attn_out),attn_w

class TimeAwareTransformerEncoderLayer(nn.Module):
    """
    一个Transformer编码器层，集成了时间感知和可选的跨注意力。
    
    它包含三个主要部分：
    1. 自注意力（Self-Attention）: 使用`TimeAwareMultiheadAttention_ContextGated`
       来处理序列内部的依赖关系，并融入时间信息。
    2. 跨注意力（Cross-Attention）: (可选) 允许序列（query）关注一个外部的
       内存源（memory），例如来自全局POI图的表示。
    3. 前馈网络（Feed-Forward Network）: 标准的MLP块。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation_fn="relu",
                 num_pairwise_time_bins=None, mha_gate_activation='sigmoid', mha_context_gate_input_type='qk',
                 has_cross_attention=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.has_cross_attention = has_cross_attention

        # Self-Attention 模块
        self.self_attn = TimeAwareMultiheadAttention_ContextGated(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            num_pairwise_time_bins=num_pairwise_time_bins,
            gate_activation=mha_gate_activation,
            context_gate_input_type=mha_context_gate_input_type)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-Attention 模块
        if self.has_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, 
                num_heads=nhead, 
                dropout=dropout, 
                batch_first=True # 确保 batch 在第一个维度
            )
            self.norm_cross = nn.LayerNorm(d_model)
            self.dropout_cross = nn.Dropout(dropout)

        # Feed-Forward 模块
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        if activation_fn == "relu":
            self.activation = F.relu
        elif activation_fn == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation_fn}")
    def forward(self,src,binned_pairwise_time_diffs,src_mask=None,src_key_padding_mask=None,cross_attention_memory=None, memory_key_padding_mask=None):
        if src.size(-1) == 0:
            return src

        # 1. Self-Attention (序列内部交互)
        # Q, K, V 都来自 src
        src2, _ = self.self_attn(src, src, src, binned_pairwise_time_diffs,
                                key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # --- 2. (可选) Cross-Attention (序列与全局图交互) ---
        if self.has_cross_attention and cross_attention_memory is not None:
            # Q 来自 src (序列表示)
            # K, V 来自 cross_attention_memory (全局图POI表示)
            src3, _ = self.cross_attn(query=src, 
                                      key=cross_attention_memory, 
                                      value=cross_attention_memory,
                                      key_padding_mask=memory_key_padding_mask)
            src = src + self.dropout_cross(src3)
            src = self.norm_cross(src)

        # 3. Feed-Forward Network 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

# ==============================================================================
# 4.1. 多关系POI图编码器
# ==============================================================================
class MultiRelationalPOIEncoder(nn.Module):
    """
    多关系POI图编码器。
    
    该模块负责对全局多关系POI图进行编码。它为每种关系类型（地理、类别、共现等）
    使用一个独立的GATv2网络栈。然后，它将所有关系特定的表示融合起来，
    得到每个POI的最终综合表示。
    """
    
    def __init__(self, 
                 num_pois,
                 num_categories_w_pad, cat_pad_idx,
                 poi_embed_dim, cat_embed_dim,
                 hidden_dim=DEFAULT_GLOBAL_GRAPH_HIDDEN_DIM,
                 num_layers=DEFAULT_GLOBAL_GRAPH_NUM_LAYERS,
                 num_heads=DEFAULT_GLOBAL_GRAPH_NUM_HEADS, 
                 dropout=DEFAULT_GLOBAL_GRAPH_DROPOUT,
                 poi_embeddings_pretrained=None,
                 cat_embeddings_pretrained=None):
        super().__init__()
        
        self.num_pois = num_pois
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = NUM_RELATION_TYPES
        
        # POI和类别嵌入
        self.poi_embedding = nn.Embedding(num_pois, poi_embed_dim)
        self.cat_embedding = nn.Embedding(num_categories_w_pad, cat_embed_dim, padding_idx=cat_pad_idx)
        
        # 加载预训练嵌入
        if poi_embeddings_pretrained is not None and poi_embeddings_pretrained.shape[0] >= num_pois:
            self.poi_embedding.weight.data[:num_pois] = poi_embeddings_pretrained[:num_pois]
        if cat_embeddings_pretrained is not None and cat_embeddings_pretrained.shape == self.cat_embedding.weight.shape:
            self.cat_embedding.weight.data.copy_(cat_embeddings_pretrained)
            
        # 节点特征编码器
        # 输入: [poi_id, category_id, lat, lon, popularity] -> hidden_dim
        self.node_feature_encoder = nn.Sequential(
            nn.Linear(poi_embed_dim + cat_embed_dim + 3, hidden_dim * 2),  # +3 for lat, lon, popularity
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 关系特定的GAT层
        self.relation_gats = nn.ModuleList()
        for rel_id in range(self.num_relations):
            rel_gats = nn.ModuleList()
            current_dim = hidden_dim
            
            for layer_idx in range(num_layers):
                is_last_layer = (layer_idx == num_layers - 1)
                out_dim = hidden_dim if is_last_layer else hidden_dim
                heads = 1 if is_last_layer else num_heads
                
                rel_gats.append(GATv2Conv(
                    current_dim, out_dim, heads=heads,
                    dropout=dropout, concat=not is_last_layer,
                    add_self_loops=True
                ))
                
                current_dim = out_dim * heads if not is_last_layer else out_dim
            
            self.relation_gats.append(rel_gats)
        
        # 关系融合层
        self.relation_fusion = nn.Sequential(
            nn.Linear(hidden_dim * self.num_relations, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 残差连接的LayerNorm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, global_graph_data, target_node_indices=None):
        """
        前向传播。支持对全图或子图进行推理。
        
        Args:
            global_graph_data (Data): PyG图数据对象。
            target_node_indices (Tensor, optional): 如果提供，则只返回这些目标节点的表示，
                                                  用于高效的批处理子图推理。
        """
        # 处理空图或无效输入的边界情况
        if global_graph_data is None or not hasattr(global_graph_data, 'x') or global_graph_data.x.size(0) == 0:
            device = next(self.parameters()).device
            if target_node_indices is not None:
                return torch.zeros(len(target_node_indices), self.hidden_dim, device=device)
            else:
                return torch.zeros(self.num_pois, self.hidden_dim, device=device)
        
        # --- GNN计算的核心逻辑 ---
        
        # 1. 构建初始节点表示
        node_features = global_graph_data.x
        poi_ids = node_features[:, 0].long()
        cat_ids = node_features[:, 1].long()
        coords_and_pop = node_features[:, 2:]
        # 获取POI和类别嵌入
        poi_embeds = self.poi_embedding(poi_ids)
        cat_embeds = self.cat_embedding(cat_ids)
        # 合并特征并编码
        combined_features = torch.cat([poi_embeds, cat_embeds, coords_and_pop], dim=-1)
        node_h = self.node_feature_encoder(combined_features)
        
        # 2. 分别通过每种关系类型的GAT
        relation_outputs = []
        for rel_id in range(self.num_relations):
            rel_mask = (global_graph_data.edge_attr == rel_id)
            if rel_mask.sum() == 0:
                relation_outputs.append(torch.zeros_like(node_h))
                continue
                
            rel_edge_index = global_graph_data.edge_index[:, rel_mask]
            
            rel_h = node_h
            for gat_layer in self.relation_gats[rel_id]:
                rel_h = gat_layer(rel_h, rel_edge_index)
                rel_h = F.elu(rel_h)
            relation_outputs.append(rel_h)
        
        # 3. 融合所有关系的表示
        if len(relation_outputs) > 0:
            concat_rel_h = torch.cat(relation_outputs, dim=-1)
            fused_h = self.relation_fusion(concat_rel_h)
            final_h = self.final_norm(node_h + fused_h)
        else:
            final_h = node_h

        # 4. 根据模式返回结果
        # 如果没有指定目标节点，说明是全图计算或推理
        if target_node_indices is None:
            # 如果输入的是全图，我们需要将表示扩展到所有POI
            # 这里的判断条件可以根据你的num_pois和padding策略微调
            if global_graph_data.num_nodes >= self.num_pois - 1: 
                full_poi_representations = torch.zeros(self.num_pois, self.hidden_dim, device=final_h.device)
                poi_ids_in_graph = global_graph_data.x[:, 0].long()
                full_poi_representations[poi_ids_in_graph] = final_h
                return full_poi_representations
            else: # 如果输入的是子图，直接返回子图所有节点的表示
                return final_h
        
        # 如果指定了目标节点，则只返回这些节点的表示
        else:
            # target_node_indices 是相对于当前输入的 graph_data 的内部节点索引
            return final_h.index_select(0, target_node_indices)

class SequenceEncoder(nn.Module):
    """
    序列编码器模块。
    
    这是模型的核心组件之一，负责将一个带有多种特征（POI, 时间, 类别, 位置等）的
    输入序列编码成一个丰富的表示。它集成了全局图信息和跨注意力机制。
    """
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout,
                 num_venues_w_pad, venue_pad_idx, v_orig_dim,
                 num_cats_w_pad, cat_pad_idx, c_orig_dim,
                 tc_orig_dim,
                 num_time_segments_w_pad, time_segment_pad_idx, ts_type_orig_dim,
                 loc_embed_dim, pop_embed_dim, max_len,
                 use_learnable_pos_enc, num_pairwise_time_bins,
                 attn_gate_activation, attn_context_gate_input_type,
                 poi_embeddings_pretrained=None,
                 cat_embeddings_pretrained=None,
                 global_poi_encoder=None,
                 enable_cross_attention=False,
                 max_cross_attn_memory_size=4096,
                 top_k_poi_indices_for_memory=None):
        super().__init__()
        self.d_model = d_model
        self.use_learnable_pos_enc = use_learnable_pos_enc
        self.global_poi_encoder = global_poi_encoder
        self.venue_pad_idx = venue_pad_idx
        self.cat_pad_idx = cat_pad_idx

        self.enable_cross_attention = enable_cross_attention
        self.max_cross_attn_memory_size = max_cross_attn_memory_size
        if top_k_poi_indices_for_memory is not None:
            # 将这个张量注册为缓冲区(buffer)，它会随模型移动到GPU，但不是模型参数
            self.register_buffer('top_k_poi_indices', top_k_poi_indices_for_memory)
        else:
            self.top_k_poi_indices = None

        # --- 嵌入层定义 ---
        self.venue_embedding = nn.Embedding(num_venues_w_pad, v_orig_dim, padding_idx=venue_pad_idx)
        self.cat_embedding = nn.Embedding(num_cats_w_pad, c_orig_dim, padding_idx=cat_pad_idx)
        
        if poi_embeddings_pretrained is not None:
            if poi_embeddings_pretrained.shape == self.venue_embedding.weight.shape:
                self.venue_embedding.weight.data.copy_(poi_embeddings_pretrained)
            else:
                print(f"SequenceEncoder: POI嵌入维度不匹配，跳过加载。")
        if cat_embeddings_pretrained is not None:
            if cat_embeddings_pretrained.shape == self.cat_embedding.weight.shape:
                self.cat_embedding.weight.data.copy_(cat_embeddings_pretrained)
            else:
                print(f"SequenceEncoder: 类别嵌入维度不匹配，跳过加载。")

        self.hour_encoding = nn.Embedding.from_pretrained(generate_cyclical_encoding_table(24, tc_orig_dim), freeze=False)
        self.time_segment_embedding = nn.Embedding(num_time_segments_w_pad, ts_type_orig_dim, padding_idx=time_segment_pad_idx)
        self.location_encoder = nn.Sequential(nn.Linear(2, loc_embed_dim * 2), nn.ReLU(), nn.Linear(loc_embed_dim * 2, loc_embed_dim))
        self.popularity_encoder = nn.Sequential(nn.Linear(1, pop_embed_dim * 2), nn.ReLU(), nn.Linear(pop_embed_dim * 2, pop_embed_dim))
        
        # --- 投影层定义 ---
        self.use_global_graph = (global_poi_encoder is not None)
        if self.use_global_graph:
            # 这个投影层既用于初始融合，也用于Cross-Attention的memory准备
            self.global_graph_proj = nn.Linear(global_poi_encoder.hidden_dim, d_model)
        
        self.venue_proj = nn.Linear(v_orig_dim, d_model)
        self.cat_proj = nn.Linear(c_orig_dim, d_model)
        self.hour_proj = nn.Linear(tc_orig_dim, d_model)
        self.ts_type_proj = nn.Linear(ts_type_orig_dim, d_model)
        self.loc_proj = nn.Linear(loc_embed_dim, d_model)
        self.pop_proj = nn.Linear(pop_embed_dim, d_model)
        
        # --- 特征融合权重 (可学习) ---
        self.w_venue = nn.Parameter(torch.ones(1))
        self.w_cat = nn.Parameter(torch.ones(1))
        self.w_hour = nn.Parameter(torch.ones(1))
        self.w_ts_type = nn.Parameter(torch.ones(1))
        self.w_loc = nn.Parameter(torch.ones(1))
        self.w_pop = nn.Parameter(torch.ones(1))
        if self.use_global_graph:
            self.w_global = nn.Parameter(torch.ones(1))
        self.fusion_layernorm = nn.LayerNorm(d_model)

        # --- 位置编码 ---
        if use_learnable_pos_enc:
            self.pos_embedding = nn.Embedding(max_len, d_model)
        else:
            self.pos_encoder_fixed = PositionalEncoding(d_model, dropout, max_len)
        self.dropout_layer = nn.Dropout(dropout)
        
        # --- Transformer Encoder 实例化 ---
        encoder_layers_list = []
        if num_encoder_layers > 0 and d_model > 0 and nhead > 0:
            for _ in range(num_encoder_layers):
                encoder_layers_list.append(TimeAwareTransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation_fn='relu',
                    num_pairwise_time_bins=num_pairwise_time_bins,
                    mha_gate_activation=attn_gate_activation,
                    mha_context_gate_input_type=attn_context_gate_input_type,
                    # --- 传递新参数，以决定是否创建Cross-Attention子模块 ---
                    has_cross_attention=self.enable_cross_attention
                ))
            self.transformer_encoder_layers = nn.ModuleList(encoder_layers_list)
            self.encoder_norm = nn.LayerNorm(d_model)
        else:
            self.transformer_encoder_layers = nn.ModuleList()
            self.encoder_norm = None

    def forward(self, venues, hours, time_segment_types, cats, lats, lons, popularities, raw_timestamps, padding_mask,
                global_poi_representations=None):
        
        B, L = venues.shape
        # 如果模型维度为0，直接返回零张量，避免后续计算出错
        if self.d_model == 0:
            return torch.zeros(B, L, 0, device=venues.device)

        # 1. 提取并编码各种输入特征
        v_emb = self.venue_embedding(venues)
        c_emb = self.cat_embedding(cats)
        h_emb = self.hour_encoding(hours)
        ts_emb = self.time_segment_embedding(time_segment_types)
        loc_feat = torch.stack([lats, lons], dim=-1)
        loc_emb = self.location_encoder(loc_feat)
        pop_emb = self.popularity_encoder(popularities.unsqueeze(-1))
        
        # 2. 初始特征融合 (将所有特征加权求和)
        fused_terms = [
            self.w_venue * F.relu(self.venue_proj(v_emb)),
            self.w_cat * F.relu(self.cat_proj(c_emb)),
            self.w_hour * F.relu(self.hour_proj(h_emb)),
            self.w_ts_type * F.relu(self.ts_type_proj(ts_emb)),
            self.w_loc * F.relu(self.loc_proj(loc_emb)),
            self.w_pop * F.relu(self.pop_proj(pop_emb))
        ]
        
        # 2.1. (关键) 初始融合全局图表示
        if self.use_global_graph and global_poi_representations is not None:
            # 从预计算的全局表示中，为序列中的每个POI提取其表示
            global_poi_reps_on_device = global_poi_representations.to(venues.device)
            global_emb_for_seq = global_poi_reps_on_device[venues]
            global_proj = self.global_graph_proj(global_emb_for_seq)
            fused_terms.append(self.w_global * F.relu(global_proj))
        
        fused_features = self.fusion_layernorm(sum(fused_terms))
        
        # 3. 添加位置编码
        if self.use_learnable_pos_enc:
            pos_indices = torch.arange(0, L, device=venues.device).unsqueeze(0).expand(B, -1)
            tf_input = self.dropout_layer(fused_features + self.pos_embedding(pos_indices))
        else:
            tf_input = self.pos_encoder_fixed(fused_features)
        
        # 4. 准备Self-Attention所需的时间偏差矩阵
        binned_pairwise_tdiffs = bin_pairwise_time_diffs_torch(
            (raw_timestamps.unsqueeze(2) - raw_timestamps.unsqueeze(1)) / 60.0,
            PAIRWISE_TIME_DIFF_BINS
        )
        
        # 5. （关键）准备Cross-Attention所需的 memory 和 memory_key_padding_mask
        cross_attn_memory = None
        memory_key_padding_mask = None # 在这个策略下，memory是密集的，不需要mask

        if self.enable_cross_attention and self.use_global_graph and global_poi_representations is not None and self.top_k_poi_indices is not None and self.max_cross_attn_memory_size > 0:
            # a. 投影全局POI表示到Transformer的d_model维度
            projected_global_reps = self.global_graph_proj(global_poi_representations.to(venues.device))
            # b. 确定要采样的memory大小
            num_available_pois = self.top_k_poi_indices.size(0)
            effective_memory_size = min(num_available_pois, self.max_cross_attn_memory_size)
            # c. 从预先排好序的热门POI索引中，取出前k个
            memory_indices = self.top_k_poi_indices[:effective_memory_size]
            # d. 使用这些索引从投影后的全局表示中采样，构建memory
            sampled_reps = projected_global_reps[memory_indices] # -> [effective_memory_size, d_model]
            # e. 将其扩展到批次大小，作为Cross-Attention的Key和Value
            cross_attn_memory = sampled_reps.unsqueeze(0).expand(B, -1, -1) # -> [B, effective_memory_size, d_model]
            
        # 6. 通过Transformer Encoder层
        current_src = tf_input
        if self.transformer_encoder_layers:
            for layer_module in self.transformer_encoder_layers:
                current_src = layer_module(
                    src=current_src, 
                    binned_pairwise_time_diffs=binned_pairwise_tdiffs, 
                    src_key_padding_mask=padding_mask,
                    cross_attention_memory=cross_attn_memory,
                    memory_key_padding_mask=memory_key_padding_mask
                )
        
        # 7. 应用最终的LayerNorm
        tf_out = current_src
        if self.encoder_norm is not None:
            tf_out = self.encoder_norm(tf_out)
            
        return tf_out

class UserRepresentationModule(nn.Module):
    """
    用户表示模块。
    
    该模块旨在为每个用户生成一个全面的、静态的表示（用户画像）。
    它结合了两种信息源：
    1. 序列信息：使用一个Transformer（`SequenceEncoder`）编码用户的完整历史轨迹。
    2. 局部图信息：使用一个GATv2网络编码用户历史轨迹形成的局部动态图。
    
    最后，将这两种表示融合起来，形成最终的用户画像。
    """
    def __init__(self, 
                 sequence_encoder,
                 num_total_venues, venue_embed_dim_gat_node_id,
                 num_total_categories_w_pad, cat_embed_dim_gat_node_cat, cat_pad_idx_gat_node,
                 loc_input_dim_gat_node, loc_embed_dim_gat_node,
                 pop_input_dim_gat_node, pop_embed_dim_gat_node,
                 transe_embed_dim,
                 num_edge_time_bins_gat_w_pad, edge_time_embed_dim_gat, edge_time_pad_idx_gat,
                 num_edge_dist_bins_gat_w_pad, edge_dist_embed_dim_gat, edge_dist_pad_idx_gat,
                 gat_hidden_dims_list, gat_num_heads_list, gat_output_dim, gat_dropout, use_gat,
                 user_rep_fusion_type, user_rep_final_dim,
                 poi_embeddings_pretrained=None,
                 cat_embeddings_pretrained=None,
                 global_poi_encoder=None  # 新增参数
                ):
        super().__init__()
        self.sequence_encoder = sequence_encoder
        self.transformer_d_model = self.sequence_encoder.d_model
        self.fusion_dropout = self.sequence_encoder.dropout_layer.p
        self.global_poi_encoder = global_poi_encoder

        self.use_gat = use_gat and GATv2Conv is not None
        self.gat_output_dim_internal = max(0, gat_output_dim) if self.use_gat else 0

        if self.use_gat:
            self.gat_node_poi_id_embedding = nn.Embedding(num_total_venues, venue_embed_dim_gat_node_id)
            self.gat_node_poi_cat_embedding = nn.Embedding(num_total_categories_w_pad, cat_embed_dim_gat_node_cat, padding_idx=cat_pad_idx_gat_node)
            if cat_embeddings_pretrained is not None and cat_embeddings_pretrained.shape[0] == num_total_categories_w_pad and cat_embeddings_pretrained.shape[1] == cat_embed_dim_gat_node_cat:
                self.gat_node_poi_cat_embedding.weight.data.copy_(cat_embeddings_pretrained)
            
            self.gat_node_poi_loc_encoder = nn.Sequential(
                nn.Linear(loc_input_dim_gat_node, max(1,loc_embed_dim_gat_node*2)), nn.ReLU(),
                nn.Linear(max(1,loc_embed_dim_gat_node*2), loc_embed_dim_gat_node))
                
            self.gat_node_poi_pop_encoder = nn.Sequential(
                nn.Linear(pop_input_dim_gat_node, max(1, pop_embed_dim_gat_node*2)), nn.ReLU(),
                nn.Linear(max(1, pop_embed_dim_gat_node*2), pop_embed_dim_gat_node)
            )

            calculated_initial_node_dim = (
                venue_embed_dim_gat_node_id +
                transe_embed_dim +
                cat_embed_dim_gat_node_cat +
                loc_embed_dim_gat_node +
                pop_embed_dim_gat_node
            )
            
            self.gat_initial_node_proj = nn.Sequential(
                nn.Linear(max(1, calculated_initial_node_dim), max(1, calculated_initial_node_dim)),
                nn.ReLU(),
                nn.LayerNorm(max(1, calculated_initial_node_dim))
            )
            current_gat_node_dim = calculated_initial_node_dim
            
            self.gat_edge_time_embedding = nn.Embedding(num_edge_time_bins_gat_w_pad, edge_time_embed_dim_gat, padding_idx=edge_time_pad_idx_gat)
            self.gat_edge_dist_embedding = nn.Embedding(num_edge_dist_bins_gat_w_pad, edge_dist_embed_dim_gat, padding_idx=edge_dist_pad_idx_gat)
            self.gat_edge_feature_dim_used = edge_time_embed_dim_gat + edge_dist_embed_dim_gat if (edge_time_embed_dim_gat + edge_dist_embed_dim_gat > 0) else None

            self.gat_layers = nn.ModuleList()
            self.gat_norms = nn.ModuleList()

            if current_gat_node_dim > 0 and gat_hidden_dims_list:
                for i, (hidden_dim_total, num_h) in enumerate(zip(gat_hidden_dims_list, gat_num_heads_list)):
                    is_last_gat_layer = (i == len(gat_hidden_dims_list) - 1)
                    gat_layer_out_ch_per_head = hidden_dim_total // num_h if num_h > 0 else hidden_dim_total
                    if gat_layer_out_ch_per_head == 0 and hidden_dim_total > 0: gat_layer_out_ch_per_head = 1
                    
                    self.gat_layers.append(GATv2Conv(
                        current_gat_node_dim, gat_layer_out_ch_per_head, heads=num_h,
                        dropout=gat_dropout, concat=(not is_last_gat_layer), add_self_loops=True,
                        edge_dim=self.gat_edge_feature_dim_used
                    ))
                    
                    if not is_last_gat_layer:
                        current_gat_node_dim = gat_layer_out_ch_per_head * num_h
                    else:
                        current_gat_node_dim = gat_layer_out_ch_per_head
                    
                    self.gat_norms.append(nn.LayerNorm(current_gat_node_dim))
            
            self.gat_output_projection = nn.Linear(max(1,current_gat_node_dim), self.gat_output_dim_internal) if current_gat_node_dim != self.gat_output_dim_internal and self.gat_output_dim_internal > 0 and current_gat_node_dim > 0 else nn.Identity()
        
        # --- 序列表示和图表示的融合模块 ---
        self.user_rep_fusion_type = user_rep_fusion_type
        self.user_rep_final_dim = user_rep_final_dim if user_rep_final_dim is not None else self.transformer_d_model
        
        fusion_input_dim_actual = self.transformer_d_model + self.gat_output_dim_internal
        fusion_input_dim_actual = max(1, fusion_input_dim_actual)

        if self.user_rep_fusion_type == 'seq_only' or not self.use_gat:
            self.user_rep_fusion_layer = nn.Linear(self.transformer_d_model, self.user_rep_final_dim) if self.transformer_d_model != self.user_rep_final_dim else nn.Identity()
        elif self.user_rep_fusion_type == 'add':
            self.proj_seq_for_add = nn.Linear(self.transformer_d_model, self.user_rep_final_dim)
            self.proj_graph_for_add = nn.Linear(self.gat_output_dim_internal, self.user_rep_final_dim) if self.gat_output_dim_internal > 0 else None
        elif self.user_rep_fusion_type == 'concat_mlp':
            self.user_rep_fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_dim_actual, max(1, fusion_input_dim_actual // 2)), nn.ReLU(),
                nn.Dropout(self.fusion_dropout),
                nn.Linear(max(1, fusion_input_dim_actual // 2), self.user_rep_final_dim))
        elif self.user_rep_fusion_type == 'gated':
            self.gate_fc_user_rep = nn.Linear(fusion_input_dim_actual, self.user_rep_final_dim)
            self.proj_seq_for_gated_fusion = nn.Linear(self.transformer_d_model, self.user_rep_final_dim)
            self.proj_graph_for_gated_fusion = nn.Linear(self.gat_output_dim_internal, self.user_rep_final_dim) if self.gat_output_dim_internal > 0 else None
        else:
            raise ValueError(f"不支持的用户画像融合类型: {self.user_rep_fusion_type}")
        
        self.init_non_encoder_weights()

    def init_non_encoder_weights(self):
        initrange = 0.1

        if self.use_gat:
            for emb in [self.gat_node_poi_id_embedding, self.gat_node_poi_cat_embedding, 
                        self.gat_edge_time_embedding, self.gat_edge_dist_embedding]:
                if emb is not None and emb.weight.requires_grad:
                    emb.weight.data.uniform_(-initrange, initrange)
                    if emb.padding_idx is not None:
                        with torch.no_grad():
                            emb.weight[emb.padding_idx].fill_(0)
            
            for encoder in [self.gat_node_poi_loc_encoder, self.gat_node_poi_pop_encoder, self.gat_initial_node_proj]:
                if hasattr(encoder, '__iter__'):
                    for layer in encoder:
                        if isinstance(layer, nn.Linear) and layer.in_features > 0 and layer.out_features > 0:
                            gain = nn.init.calculate_gain('relu' if 'ReLU' in str(layer) else 'linear')
                            nn.init.xavier_uniform_(layer.weight, gain=gain)
                            if layer.bias is not None:
                                layer.bias.data.zero_()
            
            if hasattr(self, 'gat_output_projection') and isinstance(self.gat_output_projection, nn.Linear):
                if self.gat_output_projection.in_features > 0:
                    nn.init.xavier_uniform_(self.gat_output_projection.weight)
                    if self.gat_output_projection.bias is not None:
                        self.gat_output_projection.bias.data.zero_()

        fusion_layers_to_init = []
        if self.user_rep_fusion_type == 'concat_mlp':
            fusion_layers_to_init.append(self.user_rep_fusion_layer)
        elif self.user_rep_fusion_type == 'add':
            fusion_layers_to_init.extend([self.proj_seq_for_add, self.proj_graph_for_add])
        elif self.user_rep_fusion_type == 'gated':
            fusion_layers_to_init.extend([self.gate_fc_user_rep, self.proj_seq_for_gated_fusion, self.proj_graph_for_gated_fusion])
        elif self.user_rep_fusion_type == 'seq_only':
            fusion_layers_to_init.append(self.user_rep_fusion_layer)

        for module in fusion_layers_to_init:
            if module is None: continue
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear) and layer.in_features > 0:
                        gain = nn.init.calculate_gain('relu' if 'ReLU' in str(layer) else 'linear')
                        nn.init.xavier_uniform_(layer.weight, gain=gain)
                        if layer.bias is not None: layer.bias.data.zero_()
            elif isinstance(module, nn.Linear) and module.in_features > 0:
                is_gate_fc = (module is getattr(self, 'gate_fc_user_rep', None))
                gain = nn.init.calculate_gain('sigmoid') if is_gate_fc else 1.0
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None: module.bias.data.zero_()

    def forward(self, user_history_batch_dict, global_poi_representations=None):
        venues_s = user_history_batch_dict['venues_seq']
        hours_s = user_history_batch_dict['hours_seq']
        time_segment_types_s = user_history_batch_dict['time_segment_types_seq']
        cats_s = user_history_batch_dict['cats_seq']
        lats_s = user_history_batch_dict['lats_seq']
        lons_s = user_history_batch_dict['lons_seq']
        popularities_s = user_history_batch_dict['popularities_seq']
        raw_timestamps_s = user_history_batch_dict['raw_timestamps_seq']
        padding_mask_s = user_history_batch_dict['padding_mask_seq']
        seq_lens_s = user_history_batch_dict['seq_lens']
        B_seq = venues_s.size(0)

        # 1. 获取序列表示
        tf_out = self.sequence_encoder(
            venues=venues_s, hours=hours_s, time_segment_types=time_segment_types_s,
            cats=cats_s, lats=lats_s, lons=lons_s, popularities=popularities_s,
            raw_timestamps=raw_timestamps_s, padding_mask=padding_mask_s,
            global_poi_representations=global_poi_representations  # 传递全局图表示
        )
        valid_lens_s = seq_lens_s.float().clamp(min=1).unsqueeze(1).to(tf_out.device)
        mask_expanded_s = (~padding_mask_s).unsqueeze(-1).float()
        user_rep_seq = (tf_out * mask_expanded_s).sum(dim=1) / valid_lens_s

        # 2. 获取图表示
        user_rep_graph = torch.zeros(B_seq, self.gat_output_dim_internal, device=user_rep_seq.device)
        
        if self.use_gat and 'gat_batch' in user_history_batch_dict:
            gat_batch_data = user_history_batch_dict.get('gat_batch')
            if gat_batch_data is not None and gat_batch_data.num_graphs > 0 and hasattr(gat_batch_data, 'x_node_venue_ids') and gat_batch_data.x_node_venue_ids.numel() > 0:
                
                node_id_emb = self.gat_node_poi_id_embedding(gat_batch_data.x_node_venue_ids)
                node_transe_emb = self.sequence_encoder.venue_embedding(gat_batch_data.x_node_venue_ids)
                node_cat_emb = self.gat_node_poi_cat_embedding(gat_batch_data.x_node_cat_ids)
                node_loc_emb = self.gat_node_poi_loc_encoder(gat_batch_data.x_node_locs)
                node_pop_emb = self.gat_node_poi_pop_encoder(gat_batch_data.x_node_popularity)

                initial_node_features_concat = torch.cat([
                    node_id_emb, node_transe_emb, node_cat_emb, node_loc_emb, node_pop_emb
                ], dim=-1)
                current_gat_h = self.gat_initial_node_proj(initial_node_features_concat)
                
                edge_attr_for_gat = None
                if self.gat_edge_feature_dim_used is not None and hasattr(gat_batch_data, 'edge_attr') and gat_batch_data.edge_attr.numel() > 0:
                    edge_time_emb = self.gat_edge_time_embedding(gat_batch_data.edge_attr[:, 0])
                    edge_dist_emb = self.gat_edge_dist_embedding(gat_batch_data.edge_attr[:, 1])
                    edge_attr_for_gat = torch.cat([edge_time_emb, edge_dist_emb], dim=-1)

                for i, gat_layer in enumerate(self.gat_layers):
                    h_in = current_gat_h
                    h_out = gat_layer(h_in, gat_batch_data.edge_index, edge_attr=edge_attr_for_gat)

                    if h_in.shape == h_out.shape:
                        h_out = h_in + h_out
                    
                    h_out = self.gat_norms[i](F.elu(h_out))
                    current_gat_h = h_out
                
                projected_gat_node_h = self.gat_output_projection(current_gat_h)
                graph_level_rep_batched = global_mean_pool(projected_gat_node_h, gat_batch_data.batch)
                
                if graph_level_rep_batched.size(0) == user_rep_graph.size(0):
                    user_rep_graph = graph_level_rep_batched

        # 3. 融合两种表示
        final_user_rep = None
        if self.user_rep_fusion_type == 'seq_only' or not self.use_gat:
            final_user_rep = self.user_rep_fusion_layer(user_rep_seq)
            
        elif self.user_rep_fusion_type == 'add':
            seq_proj = self.proj_seq_for_add(user_rep_seq)
            graph_proj = self.proj_graph_for_add(user_rep_graph) if self.proj_graph_for_add is not None else torch.zeros_like(seq_proj)
            final_user_rep = seq_proj + graph_proj
            
        elif self.user_rep_fusion_type == 'concat_mlp':
            concat_rep = torch.cat([user_rep_seq, user_rep_graph], dim=-1)
            final_user_rep = self.user_rep_fusion_layer(concat_rep)
            
        elif self.user_rep_fusion_type == 'gated':
            proj_seq = self.proj_seq_for_gated_fusion(user_rep_seq)
            proj_graph = self.proj_graph_for_gated_fusion(user_rep_graph) if self.proj_graph_for_gated_fusion is not None else torch.zeros_like(proj_seq)
            
            concat_rep_for_gate = torch.cat([user_rep_seq, user_rep_graph], dim=-1)
            gate_val = torch.sigmoid(self.gate_fc_user_rep(concat_rep_for_gate))
            
            final_user_rep = gate_val * proj_seq + (1 - gate_val) * proj_graph
            
        else:
            final_user_rep = user_rep_seq 

        return final_user_rep

class TrajectoryTransformer(nn.Module):
    """
    轨迹预测Transformer。
    
    这是模型的主体部分，负责进行最终的预测。它接收一个短期的、当前的
    轨迹序列，并预测下一个POI的类别。
    
    它集成了多种信息：
    - 当前轨迹的上下文信息（通过其内部的`SequenceEncoder`）。
    - 全局知识（通过`SequenceEncoder`的跨注意力从全局图中获取）。
    - 社交信息（通过与当前用户相似的其他用户的画像进行融合）。
    """
    def __init__(self,
                 sequence_encoder,
                 user_rep_dim,
                 num_categories_g,
                 pooling_strategy,
                 dropout,
                 use_similar_user_fusion, 
                 num_similar_users_k,
                 similar_user_rep_dim_scale, 
                 similar_user_aggregation_temp, 
                 similar_user_fusion_gate_type
                ):
        super().__init__()
        self.sequence_encoder = sequence_encoder
        self.d_model = self.sequence_encoder.d_model

        num_cats_w_pad = self.sequence_encoder.cat_embedding.num_embeddings
        num_venues_w_pad = self.sequence_encoder.venue_embedding.num_embeddings
        
        self.pooling_strategy = pooling_strategy
        if self.pooling_strategy == 'attention' and self.d_model > 0:
            self.attention_pooling = AttentionPooling(self.d_model)
        else:
            self.attention_pooling = None
            
        self.use_similar_user_fusion = use_similar_user_fusion and num_similar_users_k > 0
        self.num_similar_users_k = num_similar_users_k
        self.similar_user_aggregation_temp = similar_user_aggregation_temp
        self.similar_user_fusion_gate_type = similar_user_fusion_gate_type
        self.similar_user_fusion_dim = 0
        
        final_output_input_dim = self.d_model
        
        if self.use_similar_user_fusion:
            agg_sim_rep_dim = user_rep_dim
            self.similar_user_fusion_dim = max(1, int(self.d_model * similar_user_rep_dim_scale))
            
            mlp_in_dim = self.d_model + agg_sim_rep_dim
            self.fusion_with_similar_users_mlp = nn.Sequential(
                nn.Linear(mlp_in_dim, self.d_model * 2), nn.ReLU(),
                nn.Dropout(dropout), 
                nn.Linear(self.d_model * 2, self.similar_user_fusion_dim))
            
            self.similar_user_gate_fc = None
            self.similar_user_gate_mlp = None
            self.final_concat_gate_fc = None
            if self.similar_user_fusion_gate_type == 'simple_sigmoid':
                self.similar_user_gate_fc = nn.Linear(self.d_model, 1)
            elif self.similar_user_fusion_gate_type == 'contextual_mlp':
                self.similar_user_gate_mlp = nn.Sequential(
                    nn.Linear(self.d_model + agg_sim_rep_dim, self.d_model), nn.ReLU(), 
                    nn.Linear(self.d_model, 2))
            elif self.similar_user_fusion_gate_type == 'final_concat_gate':
                 self.final_concat_gate_fc = nn.Linear(self.d_model + self.similar_user_fusion_dim, 1)

            final_output_input_dim += self.similar_user_fusion_dim

        self.output_layer = nn.Linear(max(1, final_output_input_dim), num_categories_g)
        self.aux_cat_output_layer = nn.Linear(max(1, self.d_model), num_cats_w_pad)
        self.aux_venue_output_layer = nn.Linear(max(1, self.d_model), num_venues_w_pad)
        
        self.init_non_encoder_weights()

    def init_non_encoder_weights(self):
        modules_to_init = [
            self.output_layer, 
            self.aux_cat_output_layer, 
            self.aux_venue_output_layer
        ]
        if self.attention_pooling and hasattr(self.attention_pooling, 'attention_net'):
            modules_to_init.append(self.attention_pooling.attention_net)
            
        if hasattr(self, 'fusion_with_similar_users_mlp') and self.fusion_with_similar_users_mlp is not None:
            modules_to_init.append(self.fusion_with_similar_users_mlp)
        if hasattr(self, 'similar_user_gate_fc') and self.similar_user_gate_fc is not None:
            modules_to_init.append(self.similar_user_gate_fc)
        if hasattr(self, 'similar_user_gate_mlp') and self.similar_user_gate_mlp is not None:
            modules_to_init.append(self.similar_user_gate_mlp)
        if hasattr(self, 'final_concat_gate_fc') and self.final_concat_gate_fc is not None:
            modules_to_init.append(self.final_concat_gate_fc)

        for module in modules_to_init:
            if module is None: continue
            
            is_gate_fc = False
            if hasattr(self, 'similar_user_gate_fc') and module is self.similar_user_gate_fc:
                is_gate_fc = True
            if hasattr(self, 'final_concat_gate_fc') and module is self.final_concat_gate_fc:
                is_gate_fc = True

            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear) and layer.in_features > 0 and layer.out_features > 0:
                        gain = nn.init.calculate_gain('relu' if 'ReLU' in str(layer).lower() else 'linear')
                        nn.init.xavier_uniform_(layer.weight, gain=gain)
                        if layer.bias is not None: layer.bias.data.zero_()
            elif isinstance(module, nn.Linear) and module.in_features > 0 and module.out_features > 0:
                gain = nn.init.calculate_gain('sigmoid') if is_gate_fc else 1.0
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None: module.bias.data.zero_()

    def forward(self, venues, hours, time_segment_types, cats, lats, lons, popularities,
                raw_timestamps, padding_mask, seq_lens, current_user_ids_batch=None,
                all_train_user_reps_tensor=None, train_user_id_to_idx_map=None,
                global_poi_representations=None):
        
        # 1. 编码当前轨迹序列
        tf_out = self.sequence_encoder(
            venues, hours, time_segment_types, cats, lats, lons, popularities,
            raw_timestamps, padding_mask, global_poi_representations=global_poi_representations
        )

         # 2. 计算辅助任务的logits
        aux_cat_logits = self.aux_cat_output_layer(tf_out)
        # aux_venue_logits = self.aux_venue_output_layer(tf_out)
        aux_venue_logits = None # 优化：不计算venue的辅助任务
        
        # 3. 池化Transformer的输出，得到当前轨迹的表示
        pooled_output = None
        B, L, E = tf_out.shape
        
        if self.pooling_strategy == 'last':
            last_indices = (seq_lens - 1).view(-1, 1, 1).expand(-1, -1, E)
            pooled_output = tf_out.gather(1, last_indices.clamp(min=0)).squeeze(1)
        elif self.pooling_strategy == 'avg':
            valid_lens = seq_lens.float().clamp(min=1).unsqueeze(1).to(tf_out.device)
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            pooled_output = (tf_out * mask_expanded).sum(dim=1) / valid_lens
        elif self.pooling_strategy == 'max':
            masked_tf_out = tf_out.masked_fill(padding_mask.unsqueeze(-1), -1e9)
            pooled_output, _ = masked_tf_out.max(dim=1)
            pooled_output.masked_fill_(pooled_output == -1e9, 0.0)
        elif self.pooling_strategy == 'attention' and self.attention_pooling is not None:
            pooled_output = self.attention_pooling(tf_out, padding_mask)
        else:
            valid_lens = seq_lens.float().clamp(min=1).unsqueeze(1).to(tf_out.device)
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            pooled_output = (tf_out * mask_expanded).sum(dim=1) / valid_lens
        
        if pooled_output is None:
            raise ValueError("池化输出为None，请检查池化策略和输入。")

        final_representation_for_output_layer = pooled_output
        
        # 4. (关键) 融合相似用户的信息
        if self.use_similar_user_fusion:
            fused_info = torch.zeros(pooled_output.size(0), self.similar_user_fusion_dim, device=pooled_output.device)
            
            conditions_met = (
                current_user_ids_batch is not None and
                all_train_user_reps_tensor is not None and
                train_user_id_to_idx_map is not None
            )
            
            if conditions_met:
                aggregated_similar_user_reps_batch = []
                current_device = pooled_output.device
                all_train_user_reps_tensor = all_train_user_reps_tensor.to(current_device)

                for i in range(B):
                    # a. 找到当前用户的预计算画像
                    current_user_id = current_user_ids_batch[i]
                    if current_user_id not in train_user_id_to_idx_map:
                        aggregated_similar_user_reps_batch.append(torch.zeros(self.d_model, device=current_device))
                        continue
                    
                    current_user_idx = train_user_id_to_idx_map[current_user_id]
                    current_user_rep = all_train_user_reps_tensor[current_user_idx]
                    
                    # b. 计算与所有其他用户画像的余弦相似度
                    similarities = F.cosine_similarity(current_user_rep.unsqueeze(0), all_train_user_reps_tensor, dim=1)
                    similarities[current_user_idx] = -float('inf')
                    
                    actual_k_val = min(self.num_similar_users_k, len(similarities) - 1 if len(similarities) > 1 else 0)
                    if actual_k_val <= 0:
                        aggregated_similar_user_reps_batch.append(torch.zeros(self.d_model, device=current_device))
                        continue

                    # c. 找到最相似的K个用户
                    top_k_sim_scores, top_k_indices = torch.topk(similarities, k=actual_k_val)
                    sim_user_reps = all_train_user_reps_tensor[top_k_indices]
                    
                    # d. 对这K个用户的画像进行加权平均（权重由相似度决定）
                    sim_weights = F.softmax(top_k_sim_scores / self.similar_user_aggregation_temp, dim=0)
                    
                    agg_sim_rep = torch.sum(sim_user_reps * sim_weights.unsqueeze(1), dim=0)
                    aggregated_similar_user_reps_batch.append(agg_sim_rep)
                
                aggregated_similar_reps_tensor = torch.stack(aggregated_similar_user_reps_batch)
                
                # e. 将当前轨迹表示和聚合的相似用户表示融合
                combined_for_mlp = torch.cat([pooled_output, aggregated_similar_reps_tensor], dim=1)
                fused_similar_user_info = self.fusion_with_similar_users_mlp(combined_for_mlp)

                # f. 使用门控机制调整融合强度
                if self.similar_user_fusion_gate_type == 'simple_sigmoid':
                    gate = torch.sigmoid(self.similar_user_gate_fc(pooled_output))
                    fused_info = fused_similar_user_info * gate
                elif self.similar_user_fusion_gate_type == 'contextual_mlp':
                    gate_mlp_in = torch.cat([pooled_output, aggregated_similar_reps_tensor], dim=1)
                    logits2 = self.similar_user_gate_mlp(gate_mlp_in)
                    g2 = torch.sigmoid(logits2[:, 1:2])
                    fused_info = fused_similar_user_info * g2
                elif self.similar_user_fusion_gate_type == 'final_concat_gate':
                    temp_concat = torch.cat([pooled_output, fused_similar_user_info], dim=1)
                    gate = torch.sigmoid(self.final_concat_gate_fc(temp_concat.detach()))
                    fused_info = fused_similar_user_info * gate
                else:
                    fused_info = fused_similar_user_info
            
            # g. 将融合后的信息拼接到主表示上
            final_representation_for_output_layer = torch.cat([pooled_output, fused_info], dim=1)
            
            if conditions_met and self.similar_user_fusion_gate_type == 'contextual_mlp':
                if 'logits2' in locals():
                    g1 = torch.sigmoid(logits2[:, 0:1])
                    final_representation_for_output_layer[:, :self.d_model] *= g1

        # 5. 通过最终的输出层得到预测logits
        main_logits = self.output_layer(final_representation_for_output_layer)
        
        return main_logits, aux_cat_logits, aux_venue_logits

class POIRecommender(nn.Module):
    """
    总的POI推荐模型。
    
    这是一个封装类，它将所有子模块（全局图编码器、用户表示模块、轨迹Transformer）
    组合在一起，并根据一个统一的配置字典来实例化它们。
    它的`forward`方法根据不同的`mode`参数来调用相应的子模块，实现了灵活的
    模型使用方式（例如，仅计算用户画像，或进行完整的轨迹预测）。
    """
    def __init__(self,
                 config, 
                 num_venues_w_pad, venue_pad_idx,
                 num_cats_w_pad, cat_pad_idx, num_categories,
                 num_time_segments_w_pad, time_segment_pad_idx,
                 num_pairwise_time_bins, max_seq_len,
                 num_edge_time_bins_gat_w_pad, edge_time_pad_idx_gat,
                 num_edge_dist_bins_gat_w_pad, edge_dist_pad_idx_gat,
                 poi_embeds_pt=None, cat_embeds_pt=None,
                 top_k_poi_indices_for_memory=None
                 ):
        super().__init__()

        self.use_gat = config.get("use_gat", True) and (GATv2Conv is not None)
        self.use_global_graph = config.get("use_global_graph", True)  # 新增配置
        num_gat_layers = config.get("num_gat_layers", len(DEFAULT_GAT_HIDDEN_DIMS)) if self.use_gat else 0

        gat_hidden_dims_list = []
        gat_num_heads_list = []
        if self.use_gat and num_gat_layers > 0:
            gat_hidden_dims_list = config.get("gat_h_dim_list", [])
            gat_num_heads_list = config.get("gat_n_heads_list", [])

            if not gat_hidden_dims_list or len(gat_hidden_dims_list) != num_gat_layers:
                print("警告: GAT h_dim 配置与层数不匹配，将使用默认值。")
                gat_hidden_dims_list = []
                for i in range(num_gat_layers):
                    default_h_dim = DEFAULT_GAT_HIDDEN_DIMS[min(i, len(DEFAULT_GAT_HIDDEN_DIMS) - 1)]
                    gat_hidden_dims_list.append(config.get(f"gat_h_dim_l{i}", default_h_dim))
            
            if not gat_num_heads_list or len(gat_num_heads_list) != num_gat_layers:
                print("警告: GAT n_heads 配置与层数不匹配，将使用默认值。")
                gat_num_heads_list = []
                for i in range(num_gat_layers):
                    default_n_head = DEFAULT_GAT_NUM_HEADS_LIST[min(i, len(DEFAULT_GAT_NUM_HEADS_LIST) - 1)]
                    gat_num_heads_list.append(config.get(f"gat_n_heads_l{i}", default_n_head))
        
        # 1. 实例化全局POI图编码器
        self.global_poi_encoder = None
        if self.use_global_graph:
            self.global_poi_encoder = MultiRelationalPOIEncoder(
                num_pois=num_venues_w_pad,
                num_categories_w_pad=num_cats_w_pad, 
                cat_pad_idx=cat_pad_idx,
                poi_embed_dim=TRANSE_EMBED_DIM,
                cat_embed_dim=TRANSE_EMBED_DIM,
                hidden_dim=config.get("global_graph_hidden_dim", DEFAULT_GLOBAL_GRAPH_HIDDEN_DIM),
                num_layers=config.get("global_graph_num_layers", DEFAULT_GLOBAL_GRAPH_NUM_LAYERS),
                num_heads=config.get("global_graph_num_heads", DEFAULT_GLOBAL_GRAPH_NUM_HEADS),
                dropout=config.get("global_graph_dropout", DEFAULT_GLOBAL_GRAPH_DROPOUT),
                poi_embeddings_pretrained=poi_embeds_pt,
                cat_embeddings_pretrained=cat_embeds_pt
            )
        
        gat_node_id_embed_dim = config.get("gat_node_id_embed_dim", DEFAULT_GAT_NODE_ID_EMBED_DIM)
        gat_node_cat_embed_dim = config.get("gat_node_cat_embed_dim", DEFAULT_GAT_NODE_CAT_EMBED_DIM)
        user_rep_d_model = config.get("user_rep_tf_d_model", DEFAULT_USER_REP_TF_D_MODEL)
        user_rep_nhead = config.get("user_rep_tf_nhead", 4)
        user_rep_num_layers = config.get("user_rep_tf_layers", 2)
        user_rep_ff_mult = config.get("user_rep_tf_ff_multiplier", 2)
        user_rep_dropout = config.get("user_rep_tf_dropout", 0.1)
        v_orig_dim_user_rep = config.get("v_orig_dim_seq", TRANSE_EMBED_DIM)
        c_orig_dim_user_rep = config.get("c_orig_dim_seq", TRANSE_EMBED_DIM)
        traj_tf_d_model = config.get("traj_tf_d_model", 128)
        traj_tf_nhead = config.get("traj_tf_nhead", 4)
        traj_tf_num_layers = config.get("traj_tf_layers", 2)
        traj_tf_ff_mult = config.get("traj_tf_ff_multiplier", 2)
        v_orig_dim_traj = config.get("v_orig_dim_traj", TRANSE_EMBED_DIM)
        c_orig_dim_traj = config.get("c_orig_dim_traj", TRANSE_EMBED_DIM)
        shared_dropout = config.get("dropout_shared", 0.15)
        
        # 2. 实例化用于用户画像的序列编码器
        self.user_rep_sequence_encoder = SequenceEncoder(
            d_model=user_rep_d_model,
            nhead=user_rep_nhead,
            num_encoder_layers=user_rep_num_layers,
            dim_feedforward=user_rep_d_model * user_rep_ff_mult,
            dropout=user_rep_dropout,
            num_venues_w_pad=num_venues_w_pad, venue_pad_idx=venue_pad_idx, v_orig_dim=v_orig_dim_user_rep,
            num_cats_w_pad=num_cats_w_pad, cat_pad_idx=cat_pad_idx, c_orig_dim=c_orig_dim_user_rep,
            tc_orig_dim=max(2, int(user_rep_d_model * 0.1) * 2),
            num_time_segments_w_pad=num_time_segments_w_pad, time_segment_pad_idx=time_segment_pad_idx,
            ts_type_orig_dim=max(1, int(user_rep_d_model * config.get('user_rep_tf_ts_type_ratio', 0.1))),
            loc_embed_dim=config.get('user_rep_tf_loc_dim', 16),
            pop_embed_dim=config.get('user_rep_tf_pop_dim', 8),
            max_len=max_seq_len,
            use_learnable_pos_enc=config.get('user_rep_tf_lpe', True),
            num_pairwise_time_bins=num_pairwise_time_bins,
            attn_gate_activation=config.get('user_rep_tf_attn_gate_act', 'sigmoid'),
            attn_context_gate_input_type=config.get('user_rep_tf_attn_gate_type', 'qk'),
            poi_embeddings_pretrained=poi_embeds_pt,
            cat_embeddings_pretrained=cat_embeds_pt,
            global_poi_encoder=self.global_poi_encoder  # 传递全局图编码器
        )

        # 3. 实例化用户表示模块
        self.user_rep_module = UserRepresentationModule(
            sequence_encoder=self.user_rep_sequence_encoder,
            num_total_venues=num_venues_w_pad, 
            venue_embed_dim_gat_node_id=gat_node_id_embed_dim,
            num_total_categories_w_pad=num_cats_w_pad, 
            cat_embed_dim_gat_node_cat=gat_node_cat_embed_dim, 
            cat_pad_idx_gat_node=cat_pad_idx,
            loc_input_dim_gat_node=2, 
            loc_embed_dim_gat_node=config.get('gat_node_loc_embed_dim', DEFAULT_GAT_NODE_LOC_EMBED_DIM),
            pop_input_dim_gat_node=1,
            pop_embed_dim_gat_node=config.get('gat_node_pop_embed_dim', 16),
            transe_embed_dim=TRANSE_EMBED_DIM,
            num_edge_time_bins_gat_w_pad=num_edge_time_bins_gat_w_pad, 
            edge_time_embed_dim_gat=config.get('gat_edge_time_embed_dim', DEFAULT_EDGE_TIME_EMBED_DIM_GAT), 
            edge_time_pad_idx_gat=edge_time_pad_idx_gat,
            num_edge_dist_bins_gat_w_pad=num_edge_dist_bins_gat_w_pad, 
            edge_dist_embed_dim_gat=config.get('gat_edge_dist_embed_dim', DEFAULT_EDGE_DIST_EMBED_DIM_GAT), 
            edge_dist_pad_idx_gat=edge_dist_pad_idx_gat,
            gat_hidden_dims_list=gat_hidden_dims_list, 
            gat_num_heads_list=gat_num_heads_list,
            gat_output_dim=config.get('gat_output_dim', DEFAULT_GAT_OUTPUT_DIM), 
            gat_dropout=config.get('gat_dropout', DEFAULT_GAT_DROPOUT), 
            use_gat=self.use_gat,
            user_rep_fusion_type=config.get('user_rep_fusion_type', 'add') if self.use_gat else 'seq_only',
            user_rep_final_dim=traj_tf_d_model,
            poi_embeddings_pretrained=poi_embeds_pt,
            cat_embeddings_pretrained=cat_embeds_pt,
            global_poi_encoder=self.global_poi_encoder  # 传递全局图编码器
        )
        user_rep_output_dim = self.user_rep_module.user_rep_final_dim
        
        # 4. 实例化用于轨迹预测的序列编码器
        self.traj_tf_sequence_encoder = SequenceEncoder(
            d_model=traj_tf_d_model,
            nhead=traj_tf_nhead,
            num_encoder_layers=traj_tf_num_layers,
            dim_feedforward=traj_tf_d_model * traj_tf_ff_mult,
            dropout=shared_dropout,
            num_venues_w_pad=num_venues_w_pad, venue_pad_idx=venue_pad_idx, v_orig_dim=v_orig_dim_traj,
            num_cats_w_pad=num_cats_w_pad, cat_pad_idx=cat_pad_idx, c_orig_dim=c_orig_dim_traj,
            tc_orig_dim=max(2, int(traj_tf_d_model * 0.1) * 2),
            num_time_segments_w_pad=num_time_segments_w_pad, time_segment_pad_idx=time_segment_pad_idx,
            ts_type_orig_dim=max(1, int(traj_tf_d_model * config.get('traj_tf_ts_type_emb_dim_ratio', 0.1))),
            loc_embed_dim=config.get('traj_tf_loc_embed_dim', 16),
            pop_embed_dim=config.get('traj_tf_pop_embed_dim', 8),
            max_len=max_seq_len,
            use_learnable_pos_enc=config.get('traj_tf_use_learnable_pos_enc', True),
            num_pairwise_time_bins=num_pairwise_time_bins,
            attn_gate_activation=config.get('traj_tf_attn_gate_activation', 'sigmoid'),
            attn_context_gate_input_type=config.get('traj_tf_attn_context_gate_input_type', 'qk'),
            poi_embeddings_pretrained=poi_embeds_pt,
            cat_embeddings_pretrained=cat_embeds_pt,
            global_poi_encoder=self.global_poi_encoder,  # 传递全局图编码器
            enable_cross_attention=config.get('traj_tf_enable_cross_attention', False),
            max_cross_attn_memory_size=config.get('max_cross_attn_memory_size', 4096),
            top_k_poi_indices_for_memory=top_k_poi_indices_for_memory
        )
        
        # 5. 实例化最终的轨迹预测Transformer
        self.traj_transformer = TrajectoryTransformer(
            sequence_encoder=self.traj_tf_sequence_encoder,
            user_rep_dim=user_rep_output_dim,
            num_categories_g=num_categories,
            pooling_strategy=config.get('traj_tf_pooling_strategy', 'attention'),
            dropout=shared_dropout,
            use_similar_user_fusion=config.get("traj_tf_use_similar_user", True),
            num_similar_users_k=config.get("traj_tf_num_similar_k", 10),
            similar_user_rep_dim_scale=config.get("traj_tf_similar_user_scale", 0.5),
            similar_user_aggregation_temp=config.get("traj_tf_similar_user_temp", 1.0),
            similar_user_fusion_gate_type=config.get("traj_tf_similar_user_gate", 'contextual_mlp')
        )

    def forward(self, batch, mode, all_train_user_reps=None, train_user_id_to_idx=None, 
                global_graph_data=None, precomputed_global_reps=None, config=None):
        """
        模型的前向传播，根据模式选择不同的执行路径。
        
        Args:
            mode (str): 执行模式。
                - 'compute_global_reps': 高效地预计算所有POI在全局图中的表示。
                - 'user_rep': 计算并返回一批用户的画像。
                - 'trajectory': 执行完整的轨迹预测任务。
            precomputed_global_reps (Tensor, optional): 预计算好的全局POI表示，
                                                       避免在每次前向传播时重复计算。
        """
        
        model_device = next(self.parameters()).device

        if mode == 'compute_global_reps':
            # 这个模式在每个epoch开始时被调用，且应该在torch.no_grad()上下文中
            if self.global_poi_encoder is None or global_graph_data is None:
                return None
            
            # 从config获取GNN推理的批大小，如果config未提供，则使用默认值
            gnn_inference_batch_size = config.get('gnn_inference_batch_size', 2048) if config else 2048
            num_hops = self.global_poi_encoder.num_layers
            
            all_poi_reps = torch.zeros(
                self.global_poi_encoder.num_pois, # 使用编码器中定义的全量POI数量
                self.global_poi_encoder.hidden_dim,
                device=model_device
            )

            graph_cpu = global_graph_data.to('cpu')
            node_indices_loader = torch.arange(graph_cpu.num_nodes)
            
            pbar_gnn = tqdm(range(0, graph_cpu.num_nodes, gnn_inference_batch_size), desc="  GNN Inference", leave=False)
            for i in pbar_gnn:
                seed_nodes = node_indices_loader[i : i + gnn_inference_batch_size]
                
                subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_idx=seed_nodes, num_hops=num_hops,
                    edge_index=graph_cpu.edge_index, relabel_nodes=True,
                    num_nodes=graph_cpu.num_nodes
                )
                
                subgraph_data = Data(
                    x=graph_cpu.x[subset],
                    edge_index=sub_edge_index,
                    edge_attr=graph_cpu.edge_attr[edge_mask]
                ).to(model_device)
                
                target_node_indices_in_subgraph = mapping.to(model_device)
                # 调用 MultiRelationalPOIEncoder.forward
                subgraph_reps_for_seeds = self.global_poi_encoder(
                    subgraph_data, 
                    target_node_indices=target_node_indices_in_subgraph
                )
                
                all_poi_reps[seed_nodes] = subgraph_reps_for_seeds.to(model_device)
            
            return all_poi_reps
        
        # 全局表示现在直接从参数中获取
        global_poi_representations = precomputed_global_reps
        
        if mode == 'user_rep':
            return self.user_rep_module(batch, global_poi_representations=global_poi_representations)

        elif mode == 'trajectory':
            main_logits, aux_c_lg, aux_v_lg = self.traj_transformer(
                venues=batch['venues'],
                hours=batch['hours'],
                time_segment_types=batch['time_segment_types'],
                cats=batch['cats'],
                lats=batch['lats'],
                lons=batch['lons'],
                popularities=batch['popularities'],
                raw_timestamps=batch['raw_timestamps'],
                padding_mask=batch['padding_mask'],
                seq_lens=batch['seq_lens'],
                current_user_ids_batch=batch.get('user_ids'),
                all_train_user_reps_tensor=all_train_user_reps,
                train_user_id_to_idx_map=train_user_id_to_idx,
                global_poi_representations=global_poi_representations
            )
            return main_logits, aux_c_lg, aux_v_lg
            
        else:
            raise ValueError(f"未知的POIRecommender forward模式: {mode}")

# ==============================================================================
# 5. 训练与评估函数
# ==============================================================================
def update_user_representation_store(user_rep_model, user_full_history_dataloader, device, global_graph_data=None):
    """
    计算并更新全局的用户表示库。
    
    这个函数遍历所有训练用户，使用`UserRepresentationModule`为每个用户
    计算其画像，并将结果存储在一个全局张量`global_all_train_user_reps_tensor`中。
    这个张量将在主任务训练时用于相似用户融合。

    Args:
        user_rep_model (nn.Module): 用户画像模型 (UserRepresentationModule)。
        user_full_history_dataloader (DataLoader): 提供用户完整历史轨迹的加载器。
        device (torch.device): 目标计算设备 (如 'cuda:0')。
        global_graph_data (Union[Data, POIGraphCacheSystem, None]): 静态图或POI图缓存系统。
    """
    global global_all_train_user_reps_tensor, global_train_user_id_to_idx_map, \
           global_idx_to_train_user_id_map, global_train_user_ids_list_for_rep_calc

    if not global_train_user_ids_list_for_rep_calc:
        print("警告: 训练用户ID列表为空，无法更新用户表示库。")
        return

    user_rep_model.eval()
    user_rep_final_dim = user_rep_model.user_rep_final_dim
    
    if global_train_user_id_to_idx_map is None or len(global_train_user_id_to_idx_map) != len(global_train_user_ids_list_for_rep_calc):
        global_train_user_id_to_idx_map = {uid: i for i, uid in enumerate(global_train_user_ids_list_for_rep_calc)}
        global_idx_to_train_user_id_map = {i: uid for i, uid in enumerate(global_train_user_ids_list_for_rep_calc)}
    
    # 在目标设备上初始化空的表示张量
    all_user_reps_tensor_local = torch.zeros(len(global_train_user_ids_list_for_rep_calc), user_rep_final_dim, device=device)
    
    print("更新用户表示库...")
    if user_full_history_dataloader is None:
        print("警告: user_full_history_dataloader 为 None，无法动态更新用户表示。")
        if global_all_train_user_reps_tensor is None:
             global_all_train_user_reps_tensor = all_user_reps_tensor_local
        return

    # 预先处理静态图的POI表示
    static_poi_representations = None
    is_cache_system = hasattr(global_graph_data, 'get_poi_graph_for_batch')

    if user_rep_model.global_poi_encoder is not None and global_graph_data is not None:
        if is_cache_system:
            print("使用POI图缓存系统，将在推理时动态获取图表示。")
        else:
            # 这是传统的静态图，一次性计算
            print("使用静态全局图，正在预计算所有POI的表示...")
            static_graph_on_device = global_graph_data.to(device)
            with torch.no_grad():
                static_poi_representations = user_rep_model.global_poi_encoder(static_graph_on_device)
            print("静态图POI表示预计算完成。")

    # 开始计算每个用户的表示
    with torch.no_grad():
        pbar = tqdm(user_full_history_dataloader, desc="计算用户表示", leave=False)
        for batch_data in pbar:
            if batch_data is None:
                continue

            user_ids_in_b = batch_data['user_ids']
            
            # 将批次数据中的所有Tensor移动到目标设备
            for key_b in batch_data:
                if isinstance(batch_data[key_b], torch.Tensor):
                     batch_data[key_b] = batch_data[key_b].to(device)
                elif isinstance(batch_data[key_b], Data): # 处理PyG的Data或Batch对象
                     batch_data[key_b] = batch_data[key_b].to(device)

            poi_representations_for_batch = None
            if is_cache_system:
                # 使用缓存系统动态获取图表示
                try:
                    # 缓存系统返回的 graph_data 默认在 CPU 上
                    graph_data = global_graph_data.get_poi_graph_for_batch(batch_data, "user_rep_update")
                    if graph_data is not None and graph_data.num_nodes > 0:
                        # 明确将子图数据移动到目标设备
                        graph_data_on_device = graph_data.to(device)
                        # 模型和输入都在同一设备上，计算表示
                        poi_representations_for_batch = user_rep_model.global_poi_encoder(graph_data_on_device)
                except Exception as e:
                    print(f"⚠️ 在用户表示更新期间，缓存系统获取图失败: {e}")
                    poi_representations_for_batch = None
            else:
                # 使用预先计算好的静态图表示
                poi_representations_for_batch = static_poi_representations
            
            # 调用用户画像模型的前向传播
            # 模型内部的SequenceEncoder会处理设备问题，确保索引安全
            enhanced_reps_b = user_rep_model(batch_data, global_poi_representations=poi_representations_for_batch)
            
            # 将计算出的表示填充到总的张量中
            for i, uid_b in enumerate(user_ids_in_b):
                if uid_b in global_train_user_id_to_idx_map:
                    u_idx = global_train_user_id_to_idx_map[uid_b]
                    all_user_reps_tensor_local[u_idx] = enhanced_reps_b[i].detach()
    
    # 更新全局变量
    global_all_train_user_reps_tensor = all_user_reps_tensor_local
    print(f"用户表示库更新完成，包含 {len(global_train_user_ids_list_for_rep_calc)} 个用户。")

def train_epoch(
    model: POIRecommender,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader_main: DataLoader,
    user_full_history_dl_aux: DataLoader, 
    criterion_main,
    aux_cat_crit,
    aux_venue_crit,
    config: dict,
    fixed_params: dict,
    device: torch.device,
    global_graph_data=None
) -> dict:
    """
    执行一个训练轮次（epoch）。
    
    该函数包含完整的训练逻辑：
    1. 在每个epoch开始时，预计算全局POI表示（如果使用全局图）。
    2. 遍历主任务的dataloader。
    3. 对于每个批次，执行前向传播，计算总损失。总损失是主任务损失、
       辅助任务损失和对比学习损失的加权和。
    4. 使用`GradScaler`进行混合精度训练的反向传播和优化器步骤。
    5. 执行梯度裁剪。
    6. 更新学习率调度器。
    7. 记录并返回各种损失指标。
    """
    model.train()
    scaler = GradScaler(enabled=True)
    
    epoch_losses = defaultdict(float)
    total_samples = 0
    total_batches = len(dataloader_main)
    skipped_batches = 0

    aux_cat_w = config.get('aux_cat_loss_weight', 0.2)
    grad_clip = config.get('grad_clip_norm', 1.0)
    cl_weight = config.get('cl_loss_weight', 0.0)
    cl_samples_pool = fixed_params.get('contrastive_samples', [])
    num_cl_negatives = fixed_params.get('num_cl_negatives', 64)
    cl_batch_size = config.get('cl_batch_size', 2048)
    
    global_poi_representations = None
    if model.use_global_graph:
        with torch.no_grad(), autocast(enabled=True):
            global_poi_representations = model(
                batch=None, mode='compute_global_reps',
                global_graph_data=global_graph_data, config=config
            )

    pbar = tqdm(dataloader_main, desc="训练中", leave=False)
    for batch_main in pbar:
        optimizer.zero_grad(set_to_none=True)
        
        for key in batch_main:
            if isinstance(batch_main[key], torch.Tensor):
                batch_main[key] = batch_main[key].to(device, non_blocking=True)
        
        with autocast(enabled=True):
            loss_cl = torch.tensor(0.0, device=device)
            if cl_weight > 0 and cl_samples_pool and global_poi_representations is not None:
                cl_batch_indices = random.sample(range(len(cl_samples_pool)), k=min(cl_batch_size, len(cl_samples_pool)))
                anchor_indices = [cl_samples_pool[i]['anchor'] for i in cl_batch_indices]
                positive_indices = [cl_samples_pool[i]['positive'] for i in cl_batch_indices]
                negative_indices_flat = [neg for i in cl_batch_indices for neg in cl_samples_pool[i]['negatives']]
                
                anchor_embeds = F.embedding(torch.tensor(anchor_indices, device=device), global_poi_representations)
                positive_embeds = F.embedding(torch.tensor(positive_indices, device=device), global_poi_representations)
                negative_embeds = F.embedding(torch.tensor(negative_indices_flat, device=device), global_poi_representations).view(
                    len(anchor_indices), num_cl_negatives, -1
                )
                
                loss_cl = info_nce_loss(
                    anchor_embeds, positive_embeds, negative_embeds,
                    temperature=config.get('cl_temperature', 0.1)
                )

            main_lg, aux_c_lg, _ = model(
                batch=batch_main, mode='trajectory',
                all_train_user_reps=global_all_train_user_reps_tensor,
                train_user_id_to_idx=global_train_user_id_to_idx_map,
                precomputed_global_reps=global_poi_representations
            )

            loss_m_raw = criterion_main(main_lg, batch_main['target'])
            
            loss_ac_raw = torch.tensor(0.0, device=device)
            if aux_c_lg is not None and aux_cat_w > 0:
                ac_mask = (batch_main['aux_cats'].view(-1) != fixed_params['cat_pad_idx'])
                if ac_mask.any():
                    loss_ac_raw = aux_cat_crit(
                        aux_c_lg.reshape(-1, fixed_params['num_cats_w_pad'])[ac_mask],
                        batch_main['aux_cats'].view(-1)[ac_mask]
                    )

            total_loss_for_backward = loss_m_raw + (aux_cat_w * loss_ac_raw) + (cl_weight * loss_cl)

        if not torch.isfinite(total_loss_for_backward):
            print(f"\n⚠️  警告：检测到无效损失值 ({total_loss_for_backward.item()})。跳过此批次。")
            skipped_batches += 1
            # 必须清除上一个有效批次可能残留的梯度
            optimizer.zero_grad(set_to_none=True) 
            continue # 跳过此批次循环的剩余部分

        scaler.scale(total_loss_for_backward).backward()
        
        if grad_clip > 0:
            # 必须在梯度裁剪前调用 unscale_
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # --- 记录损失 ---
        bs = batch_main['target'].size(0)
        total_samples += bs
        epoch_losses['total_weighted'] += total_loss_for_backward.item() * bs
        epoch_losses['main_raw'] += loss_m_raw.item() * bs
        epoch_losses['aux_cat_raw'] += loss_ac_raw.item() * bs
        epoch_losses['cl_loss_raw'] += loss_cl.item() * bs

        pbar.set_postfix({
            'TotalL': f"{epoch_losses['total_weighted']/total_samples:.3f}",
            'MainL': f"{epoch_losses['main_raw']/total_samples:.3f}",
            'CLL': f"{epoch_losses['cl_loss_raw']/total_samples:.3f}",
            'LR': f"{optimizer.param_groups[0]['lr']:.2e}",
        })
        
    # 在 epoch 结束时报告是否跳过了批次
    if skipped_batches > 0:
        print(f"Epoch 总结：因无效损失值，跳过了 {skipped_batches}/{total_batches} 个批次。")

    return {k: v / total_samples if total_samples > 0 else 0 for k, v in epoch_losses.items()}

def evaluate(
    model: POIRecommender,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
    top_k=(1, 5, 10),
    global_graph_data=None,
    config=None
) -> Tuple[float, dict]:
    """
    在给定的数据集（验证集或测试集）上评估模型。
    
    计算并返回平均损失和多个评估指标，包括Top-k准确率（Acc@k）和
    平均倒数排名（MRR）。
    """
    model.eval()
    total_loss, total_samples = 0.0, 0
    correct_counts = {k: 0 for k in top_k}
    total_reciprocal_rank = 0.0
    max_k_val = max(top_k)
    
    current_all_user_reps_eval = global_all_train_user_reps_tensor
    current_user_id_to_idx_eval = global_train_user_id_to_idx_map

    # --- 评估前预计算全局POI表示 ---
    global_poi_representations = None
    if model.use_global_graph:
        with torch.no_grad(), autocast(enabled=True):
            # print("评估前预计算全局POI表示...")
            global_poi_representations = model(
                batch=None, mode='compute_global_reps',
                global_graph_data=global_graph_data, config=config
            )
            # print("预计算完成。")
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="评估中", leave=False)
        for batch in pbar:
            if batch is None: continue
            
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            targets = batch['target']

            with autocast(enabled=True):
                main_logits, _, _ = model(
                    batch=batch, mode='trajectory',
                    all_train_user_reps=current_all_user_reps_eval,
                    train_user_id_to_idx=current_user_id_to_idx_eval,
                    precomputed_global_reps=global_poi_representations
                )
                loss = criterion(main_logits, targets)
            
            main_logits = main_logits.float()
            _, top_indices = torch.topk(main_logits, max_k_val, dim=1) 
            
            current_batch_size = targets.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

            correct_preds_matrix = (top_indices == targets.unsqueeze(1))
            for k_val in top_k:
                correct_counts[k_val] += correct_preds_matrix[:, :k_val].any(dim=1).sum().item()
            
            sorted_indices_all = torch.argsort(main_logits, dim=1, descending=True)
            for i in range(current_batch_size):
                rank_tensor = (sorted_indices_all[i] == targets[i]).nonzero(as_tuple=True)[0]
                if rank_tensor.numel() > 0:
                    total_reciprocal_rank += 1.0 / (rank_tensor.item() + 1.0)
            
            if total_samples > 0:
                 pbar.set_postfix({'Loss': f"{total_loss / total_samples:.4f}"})

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    accuracies = {f"Acc@{k}": v / total_samples for k, v in correct_counts.items()}
    accuracies['MRR'] = total_reciprocal_rank / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, accuracies

# ==============================================================================
# 6. Optuna Objective Function
# ==============================================================================
def print_trial_config(trial_number: int, config: dict):
    """一个辅助函数，用于美观地打印出当前Optuna试验的超参数配置。"""
    print(f"\n" + "="*25 + f" Optuna Trial #{trial_number} " + "="*25)
    
    params_by_module = {
        "General": {},
        "Contrastive-Learning": {},
        "Global-POI-Graph": {},
        "UserRep-GAT": {},
        "UserRep-Transformer": {},
        "Trajectory-Transformer": {},
        "Trajectory-SimilarUser": {},
    }

    for key, value in config.items():
        if "cl_" in key or "contrastive" in key:
            params_by_module["Contrastive-Learning"][key] = value
        elif "global_graph" in key:
            params_by_module["Global-POI-Graph"][key] = value
        elif "gat_" in key:
            params_by_module["UserRep-GAT"][key] = value
        elif "user_rep_" in key:
            params_by_module["UserRep-Transformer"][key] = value
        elif "traj_tf_" in key:
            if "similar_user" in key:
                params_by_module["Trajectory-SimilarUser"][key] = value
            else:
                params_by_module["Trajectory-Transformer"][key] = value
        else:
            params_by_module["General"][key] = value

    for module_name, params in params_by_module.items():
        if not params: continue
        print(f"--- {module_name} Config ---")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"  {key:<30}: {value:.6f}")
            else:
                print(f"  {key:<30}: {value}")
                
    print("="*68 + "\n")

def create_optimizer_with_distinct_param_groups(model, lr_trial, user_rep_lr_scale_trial, 
                                               global_graph_lr_scale, wd_trial):
    """
    为模型创建优化器，并为不同的模块（全局图、用户画像、轨迹预测）
    设置不同的学习率和权重衰减。
    这允许对模型的不同部分进行差异化的训练。
    """
    # 收集所有参数及其所属模块
    all_params = set()
    param_groups = []
    
    # 1. 全局POI图编码器参数 (最高优先级，因为是新增组件)
    global_graph_params = set()
    if hasattr(model, 'global_poi_encoder') and model.global_poi_encoder is not None:
        for param in model.global_poi_encoder.parameters():
            if param.requires_grad:
                global_graph_params.add(param)
                all_params.add(param)
        
        if global_graph_params:
            param_groups.append({
                'params': list(global_graph_params),
                'lr': lr_trial * global_graph_lr_scale,
                'weight_decay': wd_trial,
                'name': 'global_poi_encoder'
            })
            print(f"  🌐 全局图编码器参数组: {len(global_graph_params)}个参数")
    
    # 2. 用户表示模块参数 (排除已经在全局图中的参数)
    user_rep_params = set()
    if hasattr(model, 'user_rep_module') and model.user_rep_module is not None:
        for param in model.user_rep_module.parameters():
            if param.requires_grad and param not in all_params:
                user_rep_params.add(param)
                all_params.add(param)
        
        if user_rep_params:
            param_groups.append({
                'params': list(user_rep_params),
                'lr': lr_trial * user_rep_lr_scale_trial,
                'weight_decay': wd_trial,
                'name': 'user_rep_module'
            })
            print(f"  👤 用户表示模块参数组: {len(user_rep_params)}个参数")
    
    # 3. 轨迹Transformer参数 (排除已经在前面组中的参数)
    traj_transformer_params = set()
    if hasattr(model, 'traj_transformer') and model.traj_transformer is not None:
        for param in model.traj_transformer.parameters():
            if param.requires_grad and param not in all_params:
                traj_transformer_params.add(param)
                all_params.add(param)
        
        if traj_transformer_params:
            param_groups.append({
                'params': list(traj_transformer_params),
                'lr': lr_trial,
                'weight_decay': wd_trial,
                'name': 'traj_transformer'
            })
            print(f"  🚀 轨迹Transformer参数组: {len(traj_transformer_params)}个参数")
    
    # 4. 其余参数 (兜底，确保所有参数都被包含)
    remaining_params = set()
    for param in model.parameters():
        if param.requires_grad and param not in all_params:
            remaining_params.add(param)
    
    if remaining_params:
        param_groups.append({
            'params': list(remaining_params),
            'lr': lr_trial,
            'weight_decay': wd_trial,
            'name': 'remaining_params'
        })
        print(f"  🔧 其余参数组: {len(remaining_params)}个参数")
    
    # 验证没有重复参数
    total_unique_params = len(all_params) + len(remaining_params)
    total_model_params = sum(1 for p in model.parameters() if p.requires_grad)
    
    print(f"  📊 参数统计: 模型总参数{total_model_params}, 分组参数{total_unique_params}")
    
    if total_unique_params != total_model_params:
        print(f"  ⚠️ 警告: 参数数量不匹配，可能存在重复或遗漏")
    
    return torch.optim.AdamW(param_groups)

def objective(trial: optuna.trial.Trial,
              train_loader_g: DataLoader,
              val_loader_g: DataLoader,
              # user_full_history_dataloader_g 被移除，因为它将在内部动态创建
              fixed_params: dict,
              poi_embeds_pt: torch.Tensor,
              cat_embeds_pt: torch.Tensor,
              global_graph_data
             ) -> float:
    """
    Optuna的目标函数。
    
    在每次试验中，此函数会：
    1. 使用`trial.suggest_*`方法从预定义的范围内采样超参数。
    2. 根据采样的超参数构建模型。
    3. 动态地创建所需的数据加载器（例如，只有在启用GAT时才创建用户历史加载器）。
    4. 运行训练和验证循环。
    5. 将验证集上的性能指标（如Acc@5）报告给Optuna。
    6. Optuna使用此返回值来指导下一次的超参数采样。
    """
    # =================================================================================
    # 1. 通用训练与正则化参数
    # =================================================================================
    lr_trial = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
    wd_trial = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    ls_trial = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.02)
    dropout_shared = trial.suggest_float("dropout_shared", 0.1, 0.4, step=0.05)
    trial.suggest_categorical("grad_clip_norm", [1.0, 1.5, 2.0])
    trial.suggest_float("aux_cat_loss_weight", 0.1, 0.5)

    # =================================================================================
    # 2. 对比学习 (Contrastive Learning) 配置 (强制启用)
    # =================================================================================
    trial.set_user_attr("use_contrastive_loss", True)
    trial.suggest_float("cl_loss_weight", 0.05, 0.5, log=True)
    trial.suggest_float("cl_temperature", 0.07, 0.2)
    trial.suggest_categorical("cl_batch_size", [2048, 4096, 8192])

    # =================================================================================
    # 3. 全局POI图核心配置 (强制启用)
    # =================================================================================
    trial.set_user_attr("use_global_graph", True)
    trial.suggest_categorical("global_graph_hidden_dim", [128, 192, 256])
    trial.suggest_int("global_graph_num_layers", 1, 3)
    trial.suggest_categorical("global_graph_num_heads", [2, 4, 8])
    trial.suggest_float("global_graph_dropout", 0.1, 0.4, step=0.05)
    trial.suggest_float("global_graph_lr_scale", 0.1, 1.0, log=True)
    trial.suggest_categorical("gnn_inference_batch_size", [2048, 4096])

    # =================================================================================
    # 4. 用户画像模块 (User Representation Module) - GAT (强制启用)
    # =================================================================================
    trial.set_user_attr("use_gat", True)
    
    trial.suggest_categorical("user_rep_fusion_type", ['add', 'concat_mlp', 'gated'])
    num_gat_layers = trial.suggest_int("num_gat_layers", 1, 3)
    
    gat_h_dim_list = [trial.suggest_categorical(f"gat_h_dim_l{i}", [32, 64, 128]) for i in range(num_gat_layers)]
    gat_n_heads_list = [trial.suggest_categorical(f"gat_n_heads_l{i}", [1, 2, 4]) for i in range(num_gat_layers)]
    trial.set_user_attr("gat_h_dim_list", gat_h_dim_list)
    trial.set_user_attr("gat_n_heads_list", gat_n_heads_list)
    
    trial.suggest_float("gat_dropout", 0.1, 0.4)
    trial.suggest_float("edge_dropout_rate", 0.0, 0.3)

    # 用户画像中的Transformer参数
    user_rep_d_model_trial = trial.suggest_categorical("user_rep_tf_d_model", [128, 192, 256])
    trial.set_user_attr("v_orig_dim_seq", TRANSE_EMBED_DIM)
    trial.set_user_attr("c_orig_dim_seq", TRANSE_EMBED_DIM)
    possible_urp_tf_nh = [h for h in [2, 4, 8] if user_rep_d_model_trial % h == 0]
    if not possible_urp_tf_nh: possible_urp_tf_nh = [1]
    trial.suggest_categorical("user_rep_tf_nhead", possible_urp_tf_nh)
    trial.suggest_int("user_rep_tf_layers", 1, 3)
    trial.suggest_float("user_rep_tf_dropout", 0.1, 0.3)
    trial.suggest_float("user_rep_lr_scale", 0.1, 1.0, log=True)

    # =================================================================================
    # 5. 轨迹预测模块 (Trajectory Transformer) - (强制启用跨注意力和相似用户)
    # =================================================================================
    trial.set_user_attr("traj_tf_enable_cross_attention", True)
    trial.set_user_attr("traj_tf_use_similar_user", True)
    
    # 轨迹Transformer核心参数
    traj_tf_d_model_trial = trial.suggest_categorical("traj_tf_d_model", [192, 256, 384])
    trial.set_user_attr("v_orig_dim_traj", TRANSE_EMBED_DIM)
    trial.set_user_attr("c_orig_dim_traj", TRANSE_EMBED_DIM)
    possible_traj_tf_nh = [h for h in [2, 4, 8] if traj_tf_d_model_trial % h == 0]
    if not possible_traj_tf_nh: possible_traj_tf_nh = [1]
    trial.suggest_categorical("traj_tf_nhead", possible_traj_tf_nh)
    trial.suggest_int("traj_tf_layers", 2, 4)
    trial.suggest_categorical("traj_tf_pooling_strategy", ['avg', 'attention', 'max', 'last'])
    trial.suggest_categorical("max_cross_attn_memory_size", [2048, 4096])

    # 相似用户融合参数
    trial.suggest_int("traj_tf_num_similar_k", 5, 20)
    trial.suggest_float("traj_tf_similar_user_scale", 0.25, 1.0)
    trial.suggest_float("traj_tf_similar_user_temp", 0.5, 2.0)
    trial.suggest_categorical("traj_tf_similar_user_gate", ['simple_sigmoid', 'contextual_mlp', 'final_concat_gate'])
    
    # =================================================================================
    # 6. 配置整合与模型实例化
    # =================================================================================
    config = {**trial.params, **trial.user_attrs}
    print_trial_config(trial.number, config)

    model_fixed_params = fixed_params.copy()
    
    df_train_for_user_rep = model_fixed_params.pop('df_train_for_user_rep', None)
    contrastive_samples = model_fixed_params.pop('contrastive_samples', None)
    num_cl_negatives = model_fixed_params.pop('num_cl_negatives', None)
    model_fixed_params.pop('venue_map', None)
    model_fixed_params.pop('category_map', None)
    
    model = POIRecommender(
        config=config,
        **model_fixed_params,
        poi_embeds_pt=poi_embeds_pt,
        cat_embeds_pt=cat_embeds_pt
    ).to(DEVICE)

    # =================================================================================
    # 7. 动态创建用户历史数据加载器
    # =================================================================================
    user_rep_dl = None
    needs_user_rep_dl = config.get("use_gat", False) or config.get("traj_tf_use_similar_user", False)
    
    if needs_user_rep_dl and df_train_for_user_rep is not None:
        print("  动态创建用户画像 DataLoader...")
        user_history_ds = UserFullHistoryDataset(
            df_train_for_user_rep.groupby('user_id'), 
            fixed_params['venue_map'], fixed_params['category_map'], fixed_params['max_seq_len'],
            fixed_params['venue_pad_idx'], fixed_params['cat_pad_idx'],
            EDGE_TIME_DIFF_BINS_GAT, EDGE_DIST_BINS_GAT, global_distance_lookup,
            is_train=True, 
            edge_dropout_rate=config.get('edge_dropout_rate', 0.0)
        )
        if len(user_history_ds) > 0:
            collate_fn_user_rep = partial(user_full_history_gat_collate_fn, 
                                          venue_pad_idx_g=fixed_params['venue_pad_idx'], 
                                          cat_pad_idx_g=fixed_params['cat_pad_idx'], 
                                          time_segment_pad_idx_g=TIME_SEGMENT_PAD_IDX)
            user_rep_dl = DataLoader(user_history_ds, batch_size=BATCH_SIZE, shuffle=True, 
                                     collate_fn=collate_fn_user_rep, num_workers=4, pin_memory=True)
            print(f"  用户画像数据集样本数: {len(user_history_ds)}")
        else:
            print("  警告: 用户画像数据集为空，相关功能可能无法正常工作。")

    # =================================================================================
    # 8. 预计算用户表示
    # =================================================================================
    if config.get("traj_tf_use_similar_user", False):
        if user_rep_dl:
            print("  预计算所有训练用户的表示（用于相似用户融合）...")
            update_user_representation_store(
                user_rep_model=model.user_rep_module,
                user_full_history_dataloader=user_rep_dl,
                device=DEVICE,
                global_graph_data=global_graph_data
            )
            print("  用户表示预计算完成。")
        else:
            print("  警告: 相似用户融合已启用，但无法创建用户画像数据加载器。")

    # =================================================================================
    # 9. 损失、优化器和调度器
    # =================================================================================
    criterion_main = nn.CrossEntropyLoss(label_smoothing=ls_trial).to(DEVICE)
    aux_cat_crit = nn.CrossEntropyLoss(label_smoothing=ls_trial, ignore_index=fixed_params['cat_pad_idx']).to(DEVICE)
    aux_venue_crit = nn.CrossEntropyLoss(label_smoothing=ls_trial, ignore_index=fixed_params['venue_pad_idx']).to(DEVICE)
    
    optimizer = create_optimizer_with_distinct_param_groups(
        model=model, lr_trial=lr_trial,
        user_rep_lr_scale_trial=config.get('user_rep_lr_scale', 0.5),
        global_graph_lr_scale=config.get('global_graph_lr_scale', 0.5),
        wd_trial=wd_trial
    )
    
    num_training_steps_trial = len(train_loader_g) * EPOCHS_PER_TRIAL
    if num_training_steps_trial == 0:
        raise optuna.exceptions.TrialPruned("训练数据加载器为空。")
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, num_training_steps_trial)

    # =================================================================================
    # 10. 训练与评估循环
    # =================================================================================
    best_val_acc5_trial = 0.0
    epochs_no_improve = 0
    for epoch in range(1, EPOCHS_PER_TRIAL + 1):
        epoch_start_time = time.time()
        
        train_loss_metrics = train_epoch(
            model=model, optimizer=optimizer, scheduler=scheduler,
            dataloader_main=train_loader_g,
            user_full_history_dl_aux=user_rep_dl,
            criterion_main=criterion_main, aux_cat_crit=aux_cat_crit, aux_venue_crit=aux_venue_crit,
            config=config, 
            fixed_params={**fixed_params, 'contrastive_samples': contrastive_samples, 'num_cl_negatives': num_cl_negatives},
            device=DEVICE,
            global_graph_data=global_graph_data
        )
        
        val_loss, val_metrics = evaluate(
            model, val_loader_g, criterion_main, DEVICE, top_k=(1, 5, 10), 
            global_graph_data=global_graph_data, config=config
        )
        
        current_val_acc5 = val_metrics.get("Acc@5", 0.0)
        
        epoch_duration_secs = time.time() - epoch_start_time
        epoch_mins, epoch_secs_rem = divmod(epoch_duration_secs, 60)
        
        print(f"  T{trial.number}-E{epoch:02}:Time:{int(epoch_mins)}m{epoch_secs_rem:.0f}s | "
              f"TrL(M):{train_loss_metrics.get('main_raw', 0):.3f} | "
              f"TrL(CL):{train_loss_metrics.get('cl_loss_raw', 0):.3f} | "
              f"VaL:{val_loss:.3f} | "
              f"Acc@1:{val_metrics.get('Acc@1', 0)*100:.2f}% | "
              f"Acc@5:{current_val_acc5*100:.2f}% | "
              f"Acc@10:{val_metrics.get('Acc@10', 0)*100:.2f}% | "
              f"MRR:{val_metrics.get('MRR', 0):.4f}")

        trial.report(current_val_acc5, epoch)
        if current_val_acc5 > best_val_acc5_trial:
            best_val_acc5_trial = current_val_acc5
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if trial.should_prune():
            print(f"  Trial {trial.number} pruned.")
            raise optuna.exceptions.TrialPruned()
            
        if epochs_no_improve >= PATIENCE:
            print(f"  Trial {trial.number} early stopped due to no improvement.")
            break
            
    return best_val_acc5_trial

# ==============================================================================
# 7. 主执行流程
# ==============================================================================
def main():
    """
    程序的主入口点。
    
    它按顺序执行整个工作流程：
    1. 设置环境和随机种子。
    2. 加载和预处理数据。
    3. 构建全局POI图和对比学习样本。
    4. 进行TransE预训练或加载已有的嵌入。
    5. 划分数据集并创建数据加载器。
    6. 启动Optuna进行超参数搜索。
    7. 在搜索结束后，使用找到的最佳参数来训练最终的模型。
    8. 在测试集上评估最终模型的性能。
    """
    global num_categories, cat_pad_idx, num_categories_with_pad, \
           num_venues, venue_pad_idx, num_venues_with_pad, \
           venue_map, category_map, global_distance_lookup, \
           NUM_PAIRWISE_TIME_DIFF_BINS, NUM_TIME_SEGMENTS_W_PAD, TIME_SEGMENT_PAD_IDX, \
           global_train_user_ids_list_for_rep_calc, \
           global_all_train_user_reps_tensor, global_train_user_id_to_idx_map, global_idx_to_train_user_id_map, \
           global_poi_graph_data

    set_seed(SEED)
    print(f"使用设备: {DEVICE}")

    # 数据加载与预处理
    df_proc = load_and_preprocess_data(DATA_PATH)
    global_distance_lookup = load_precomputed_distances(DISTANCE_DATA_PATH)
    if global_distance_lookup is None:
        print("警告: 未能加载距离数据，GAT的距离边属性将不可用。")

    # 数据集划分与加载器创建
    train_ids, val_ids, test_ids = split_users(df_proc, VAL_USER_RATIO, TEST_USER_RATIO)
    global_train_user_ids_list_for_rep_calc = list(train_ids)
    df_tr = df_proc[df_proc['user_id'].isin(train_ids)]
    df_v = df_proc[df_proc['user_id'].isin(val_ids)]
    df_te = df_proc[df_proc['user_id'].isin(test_ids)]
    grp_tr_main = df_tr.groupby('user_id')
    grp_v_main = df_v.groupby('user_id')

    print("创建主任务训练/验证 Datasets & DataLoaders...")
    train_ds_main = TrajectoryDataset(
        grp_tr_main, venue_map, MAX_SEQ_LENGTH, 
        venue_pad_idx, cat_pad_idx, num_categories,
        masking_ratio=DATA_AUG_MASKING_RATIO,
        is_train=True
    )
    val_ds_main = TrajectoryDataset(
        grp_v_main, venue_map, MAX_SEQ_LENGTH, 
        venue_pad_idx, cat_pad_idx, num_categories,
        masking_ratio=0.0,
        is_train=False
    )
    print(f"主任务训练集样本: {len(train_ds_main)}, 验证集样本: {len(val_ds_main)}")
    
    collate_fn_main = partial(main_task_collate_fn, venue_pad_idx_g=venue_pad_idx, cat_pad_idx_g=cat_pad_idx, time_segment_pad_idx_g=TIME_SEGMENT_PAD_IDX)
    train_loader_main = DataLoader(train_ds_main, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_main, num_workers=4, pin_memory=True, drop_last=True)
    val_loader_main = DataLoader(val_ds_main, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_main, num_workers=4, pin_memory=True)

    # 准备 Cross-Attention Memory 所需的热门POI索引
    print("\n--- 准备热门POI索引以优化Cross-Attention ---")
    poi_counts = df_tr['geo_id'].value_counts()
    sorted_popular_poi_str_ids = poi_counts.index.tolist()
    sorted_popular_poi_indices = [
        venue_map[poi_id] for poi_id in sorted_popular_poi_str_ids if poi_id in venue_map
    ]
    top_k_poi_indices_for_memory = torch.tensor(sorted_popular_poi_indices, dtype=torch.long)
    print(f"  已生成热门POI索引，总计 {len(top_k_poi_indices_for_memory)} 个POI。")

    # 构建静态全局图
    print("\n" + "="*50)
    print("🚀 构建静态的、完整的全局POI图 (在CPU上)...")
    print("="*50)
    graph_builder = GlobalPOIGraphBuilder(venue_map, category_map, global_distance_lookup, df_tr)
    static_global_graph = graph_builder.build_global_graph()
    global_poi_graph_data = static_global_graph
    if global_poi_graph_data is not None and global_poi_graph_data.num_nodes > 0:
        print("✅ 静态全局POI图构建成功!")
        print(f"   📊 节点数 (POIs): {global_poi_graph_data.num_nodes}")
        print(f"   🔗 边数: {global_poi_graph_data.num_edges}")
    else:
        print("❌ 静态全局POI图构建失败，将禁用图功能。")

    # 为对比学习准备正负样本
    print("\n--- 为对比学习准备正负样本 ---")
    NUM_POSITIVE_SAMPLES = 1
    NUM_NEGATIVE_SAMPLES = 64
    category_to_pois = defaultdict(list)
    for i in range(global_poi_graph_data.num_nodes):
        cat_id = int(global_poi_graph_data.x[i, 1].item())
        if cat_id != cat_pad_idx:
            category_to_pois[cat_id].append(i)
    contrastive_samples = []
    all_poi_indices = list(range(global_poi_graph_data.num_nodes))
    pbar_cl = tqdm(all_poi_indices, desc="生成对比学习样本")
    for anchor_poi_idx in pbar_cl:
        anchor_cat_id = int(global_poi_graph_data.x[anchor_poi_idx, 1].item())
        if anchor_cat_id == cat_pad_idx or len(category_to_pois.get(anchor_cat_id, [])) < 2:
            continue
        positive_candidates = category_to_pois[anchor_cat_id]
        positive_poi_idx = anchor_poi_idx
        while positive_poi_idx == anchor_poi_idx:
            positive_poi_idx = random.choice(positive_candidates)
        negative_poi_indices = []
        num_negs_added = 0
        while num_negs_added < NUM_NEGATIVE_SAMPLES:
            negative_candidate = random.choice(all_poi_indices)
            candidate_cat_id = int(global_poi_graph_data.x[negative_candidate, 1].item())
            if (negative_candidate != anchor_poi_idx and 
                negative_candidate != positive_poi_idx and
                candidate_cat_id != anchor_cat_id):
                negative_poi_indices.append(negative_candidate)
                num_negs_added += 1
        contrastive_samples.append({
            'anchor': anchor_poi_idx, 'positive': positive_poi_idx, 'negatives': negative_poi_indices
        })
    print(f"  已生成 {len(contrastive_samples)} 组对比学习样本。")

    # TransE 预训练或加载
    poi_embeds_pt, cat_embeds_pt = None, None
    if DO_TRANSE_PRETRAINING:
        if os.path.exists(POI_EMBED_PATH) and os.path.exists(CAT_EMBED_PATH):
            print("发现已有的预训练嵌入文件，直接加载。")
            poi_embeds_pt = torch.load(POI_EMBED_PATH, map_location=DEVICE)
            cat_embeds_pt = torch.load(CAT_EMBED_PATH, map_location=DEVICE)
            if poi_embeds_pt.shape[1] != TRANSE_EMBED_DIM or cat_embeds_pt.shape[1] != TRANSE_EMBED_DIM:
                 print(f"警告: 已保存的嵌入维度与配置({TRANSE_EMBED_DIM})不符。将重新进行预训练。")
                 poi_embeds_pt, cat_embeds_pt = None, None
        
        if poi_embeds_pt is None:
            print("未发现预训练嵌入或维度不匹配，开始TransE预训练流程...")
            transe_triplets, _, _, num_entities, num_relations = prepare_transe_data(
                df_tr, user_map, venue_map, category_map)
            pre_trained_entity_embeddings = run_transe_pretraining(
                transe_triplets, num_entities, num_relations, 
                embed_dim=TRANSE_EMBED_DIM, device=DEVICE, epochs=TRANSE_EPOCHS, 
                batch_size=TRANSE_BATCH_SIZE, lr=TRANSE_LR, margin=TRANSE_MARGIN, neg_ratio=TRANSE_NEG_RATIO)
            _, poi_embeds_pt, cat_embeds_pt = extract_and_save_embeddings(
                pre_trained_entity_embeddings, user_map, venue_map, category_map,
                len(user_map), num_venues_with_pad, num_categories_with_pad,
                TRANSE_EMBED_DIM, USER_EMBED_PATH, POI_EMBED_PATH, CAT_EMBED_PATH)
            poi_embeds_pt, cat_embeds_pt = poi_embeds_pt.to(DEVICE), cat_embeds_pt.to(DEVICE)
    else:
        print("跳过TransE预训练。")

    

    # Optuna 超参数搜索
    study_name = "transformer_poi_cat_study"
    storage_path = os.path.join(MODEL_SAVE_DIR, f"{study_name}.db")
    storage = f"sqlite:///{storage_path}"
    print(f"\nOptuna 搜索: {study_name}, 存储: {storage}")
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction="maximize", pruner=pruner)

    fixed_params = {
        # 模型构造相关
        'num_venues_w_pad': num_venues_with_pad, 'venue_pad_idx': venue_pad_idx,
        'num_cats_w_pad': num_categories_with_pad, 'cat_pad_idx': cat_pad_idx, 'num_categories': num_categories,
        'num_time_segments_w_pad': NUM_TIME_SEGMENTS_W_PAD, 'time_segment_pad_idx': TIME_SEGMENT_PAD_IDX,
        'num_pairwise_time_bins': NUM_PAIRWISE_TIME_DIFF_BINS, 'max_seq_len': MAX_SEQ_LENGTH,
        'num_edge_time_bins_gat_w_pad': NUM_EDGE_TIME_BINS_W_PAD_GAT, 'edge_time_pad_idx_gat': EDGE_TIME_BIN_PAD_IDX_GAT,
        'num_edge_dist_bins_gat_w_pad': NUM_EDGE_DIST_BINS_W_PAD_GAT, 'edge_dist_pad_idx_gat': EDGE_DIST_BIN_PAD_IDX_GAT,
        'top_k_poi_indices_for_memory': top_k_poi_indices_for_memory,
        
        # 数据集和样本相关
        'df_train_for_user_rep': df_tr, 
        'contrastive_samples': contrastive_samples, 
        'num_cl_negatives': NUM_NEGATIVE_SAMPLES,
        
        'venue_map': venue_map,
        'category_map': category_map
    }
    
    objective_with_data = partial(objective,
                                train_loader_g=train_loader_main, 
                                val_loader_g=val_loader_main,
                                fixed_params=fixed_params,
                                poi_embeds_pt=poi_embeds_pt, 
                                cat_embeds_pt=cat_embeds_pt,
                                global_graph_data=global_poi_graph_data)

    try:
        study.optimize(objective_with_data, n_trials=OPTUNA_N_TRIALS, timeout=3600*72)
    except KeyboardInterrupt:
        print("\nOptuna 搜索被用户中断。")
    except Exception as e:
        print(f"\nOptuna 优化中发生错误: {e}")
        import traceback; traceback.print_exc()
        
    print("\nOptuna 搜索完成。")

    # 使用最佳参数进行最终训练
    if not study.best_trial:
        print("Optuna未找到成功的Trial，无法进行最终训练。")
        return

    print("\n" + "="*30 + " 最终模型训练 " + "="*30)
    best_config = {**study.best_trial.params, **study.best_trial.user_attrs}
    print("加载 Optuna 最佳超参数:")
    print_trial_config("BEST", best_config)

    model_fixed_params = fixed_params.copy()

    model_fixed_params.pop('df_train_for_user_rep', None)
    model_fixed_params.pop('contrastive_samples', None)
    model_fixed_params.pop('num_cl_negatives', None)
    model_fixed_params.pop('venue_map', None)
    model_fixed_params.pop('category_map', None)

    final_model = POIRecommender(config=best_config, **model_fixed_params,
                                 poi_embeds_pt=poi_embeds_pt, cat_embeds_pt=cat_embeds_pt).to(DEVICE)
    
    final_user_rep_dl = None
    needs_user_rep_dl = best_config.get("use_gat", False) or best_config.get("traj_tf_use_similar_user", False)
    if needs_user_rep_dl and global_train_user_ids_list_for_rep_calc:
        print("\n最佳配置需要用户画像数据，正在创建 DataLoader...")
        final_edge_dropout_rate = best_config.get('edge_dropout_rate', 0.0)
        final_user_history_ds = UserFullHistoryDataset(
            df_tr.groupby('user_id'), venue_map, category_map, MAX_SEQ_LENGTH,
            venue_pad_idx, cat_pad_idx,
            EDGE_TIME_DIFF_BINS_GAT, EDGE_DIST_BINS_GAT, global_distance_lookup,
            is_train=True, edge_dropout_rate=final_edge_dropout_rate
        )
        if len(final_user_history_ds) > 0:
            collate_fn_user_rep = partial(user_full_history_gat_collate_fn, venue_pad_idx_g=venue_pad_idx, cat_pad_idx_g=cat_pad_idx, time_segment_pad_idx_g=TIME_SEGMENT_PAD_IDX)
            final_user_rep_dl = DataLoader(final_user_history_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_user_rep, num_workers=4, pin_memory=True)
            print(f"  最终训练的用户画像数据集样本数: {len(final_user_history_ds)}")
    else:
        print("\n最佳配置不使用GAT或相似用户融合，跳过用户画像 DataLoader 创建。")

    if best_config.get("traj_tf_use_similar_user", False):
        if final_user_rep_dl is not None:
            update_user_representation_store(final_model.user_rep_module, final_user_rep_dl, DEVICE, global_poi_graph_data)
        else:
            print("\n警告: 相似用户融合已启用，但无法创建用户画像数据加载器。")
    
    final_lr = best_config.get('learning_rate', 1e-4)
    final_wd = best_config.get('weight_decay', 1e-5)
    final_ls = best_config.get('label_smoothing', 0.1)
    final_user_rep_lr_scale = best_config.get('user_rep_lr_scale', 0.5)
    final_global_graph_lr_scale = best_config.get('global_graph_lr_scale', 0.5)

    print("\n创建最终模型的优化器...")
    final_optimizer = create_optimizer_with_distinct_param_groups(
        model=final_model, lr_trial=final_lr,
        user_rep_lr_scale_trial=final_user_rep_lr_scale,
        global_graph_lr_scale=final_global_graph_lr_scale, wd_trial=final_wd
    )
    print("最终模型优化器创建完成！")

    final_epochs = 50
    final_patience = 5
    final_num_steps = len(train_loader_main) * final_epochs
    final_scheduler = get_cosine_schedule_with_warmup(final_optimizer, WARMUP_STEPS, final_num_steps)
    final_crit_main = nn.CrossEntropyLoss(label_smoothing=final_ls).to(DEVICE)
    final_aux_c_crit = nn.CrossEntropyLoss(label_smoothing=final_ls, ignore_index=cat_pad_idx).to(DEVICE)
    final_aux_v_crit = nn.CrossEntropyLoss(label_smoothing=final_ls, ignore_index=venue_pad_idx).to(DEVICE)

    best_final_acc5 = 0.0
    epochs_no_improve_final = 0
    best_model_state = None

    print(f"\n--- 开始最终模型训练 (Epochs: {final_epochs}, Patience: {final_patience}) ---")
    print(f"🌐 全局POI图: {'启用' if final_model.use_global_graph else '禁用'}")
    print(f"🔗 用户局部GAT: {'启用' if best_config.get('use_gat', False) else '禁用'}")
    print(f"🤝 跨注意力融合: {'启用' if best_config.get('traj_tf_enable_cross_attention', False) else '禁用'}")
    print(f"👥 相似用户融合: {'启用' if best_config.get('traj_tf_use_similar_user', False) else '禁用'}")
    
    for ep in range(1, final_epochs + 1):
        epoch_start_time = time.time()
        
        train_metrics = train_epoch(
            final_model, final_optimizer, final_scheduler, train_loader_main,
            final_user_rep_dl, final_crit_main, final_aux_c_crit, final_aux_v_crit,
            best_config, fixed_params, DEVICE, global_graph_data=global_poi_graph_data
        )
        
        val_loss, val_metrics = evaluate(
            final_model, val_loader_main, final_crit_main, DEVICE,
            top_k=(1, 5, 10), global_graph_data=global_poi_graph_data, config=best_config
        )
        
        epoch_duration_secs = time.time() - epoch_start_time
        epoch_mins, epoch_secs_rem = divmod(epoch_duration_secs, 60)
        
        print(f"FinalEp {ep:02}/{final_epochs}: Time:{int(epoch_mins)}m{epoch_secs_rem:.1f}s | "
              f"TrL(M):{train_metrics.get('main_raw',0):.3f} | TrL(CL):{train_metrics.get('cl_loss_raw', 0):.3f} | "
              f"VaL:{val_loss:.3f} | "
              f"Acc@1:{val_metrics.get('Acc@1', 0)*100:.2f}% | "
              f"Acc@5:{val_metrics.get('Acc@5', 0)*100:.2f}% | "
              f"Acc@10:{val_metrics.get('Acc@10', 0)*100:.2f}% | "
              f"MRR:{val_metrics.get('MRR', 0):.4f}")

        current_val_acc5 = val_metrics.get('Acc@5', 0.0)
        if current_val_acc5 > best_final_acc5:
            best_final_acc5 = current_val_acc5
            epochs_no_improve_final = 0
            best_model_state = copy.deepcopy(final_model.state_dict())
            torch.save(best_model_state, BEST_MODEL_SAVE_PATH)
            print(f"    ✨ Best Val Acc@5 improved to {best_final_acc5*100:.2f}%. Model saved.")
        else:
            epochs_no_improve_final += 1
            print(f"  Val Acc@5 did not improve for {epochs_no_improve_final} epoch(s). Best: {best_final_acc5*100:.2f}%")
        
        if epochs_no_improve_final >= final_patience:
            print(f"\nEarly stopping at epoch {ep}.")
            break

    # 在测试集上评估最终模型
    print("\n" + "="*30 + " 最终模型评估 " + "="*30)
    if best_model_state:
        print(f"加载最佳模型 (Val Acc@5: {best_final_acc5*100:.2f}%) 进行测试...")
        final_model.load_state_dict(best_model_state)
        
        if not test_ids or len(df_te) == 0:
            print("警告: 测试集为空，跳过最终评估。")
        else:
            grp_te = df_te.groupby('user_id')
            test_ds = TrajectoryDataset(
                grp_te, venue_map, MAX_SEQ_LENGTH, 
                venue_pad_idx, cat_pad_idx, num_categories,
                masking_ratio=0.0,
                is_train=False
            )
            if len(test_ds) > 0:
                print(f"测试集样本数: {len(test_ds)}")
                test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_main, num_workers=4, pin_memory=True)
                test_loss, test_metrics = evaluate(
                    final_model, test_dl, final_crit_main, DEVICE, 
                    global_graph_data=global_poi_graph_data, config=best_config
                )
                
                print("\n--- 测试集评估结果 ---")
                print(f"  Loss: {test_loss:.4f}")
                for name, val_metric in test_metrics.items():
                    print(f"  {name}: {val_metric*100 if 'Acc@' in name else val_metric:.4f}{'%' if 'Acc@' in name else ''}")
                
            else:
                print("警告: 创建的测试数据集为空。")
    else:
        print("最终训练未能产出最佳模型，跳过测试。")

def find_best_head_size(d_model, possible_heads=[8, 4, 2, 1]):
    """辅助函数，为给定的d_model找到一个有效的注意力头数。"""
    for h in possible_heads:
        if d_model % h == 0:
            return h
    return 1 # 备选方案

def run_sensitivity_analysis(
    train_loader_g, val_loader_g, model_init_params, training_params,
    poi_embeds_pt, cat_embeds_pt, global_graph_data
):
    """
    对关键超参数 d, K, 和 beta 运行系统性分析。
    """
    print("\n" + "="*30 + " 开始超参数敏感性分析 " + "="*30)

    # ==============================================================================
    # 1. 为所有运行定义一个坚实的基础配置
    # ==============================================================================
    BASE_CONFIG = {
        # 通用参数
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'dropout_shared': 0.2,
        'grad_clip_norm': 1.0,
        'aux_cat_loss_weight': 0.2,

        # 对比学习 (β 是目标参数，这里设置一个默认值)
        'use_contrastive_loss': True,
        'cl_loss_weight': 0.1,  # 这是我们的 β
        'cl_temperature': 0.1,
        'cl_batch_size': 4096,

        # 全局POI图
        'use_global_graph': True,
        'global_graph_hidden_dim': 128, # 这是我们的 d 的一部分
        'global_graph_num_layers': 2,
        'global_graph_num_heads': 4,
        'global_graph_dropout': 0.2,
        'global_graph_lr_scale': 0.5,
        'gnn_inference_batch_size': 4096,

        # 用户画像 GAT
        'use_gat': True,
        'user_rep_fusion_type': 'add',
        'num_gat_layers': 2,
        'gat_h_dim_list': [64, 64],
        'gat_n_heads_list': [4, 1],
        'gat_dropout': 0.2,
        'edge_dropout_rate': 0.1,

        # 用户画像 Transformer
        'user_rep_tf_d_model': 128, # 这是我们的 d 的一部分
        'user_rep_tf_nhead': 4,
        'user_rep_tf_layers': 2,
        'user_rep_tf_dropout': 0.2,
        'user_rep_lr_scale': 0.5,

        # 轨迹预测 Transformer (d 和 K 是目标参数)
        'traj_tf_enable_cross_attention': True,
        'traj_tf_d_model': 128, # 这是我们的 d 的一部分
        'traj_tf_nhead': 4,
        'traj_tf_layers': 3,
        'traj_tf_pooling_strategy': 'attention',
        'max_cross_attn_memory_size': 4096,

        # 相似用户融合
        'traj_tf_use_similar_user': True,
        'traj_tf_num_similar_k': 10, # 这是我们的 K
        'traj_tf_similar_user_scale': 0.5,
        'traj_tf_similar_user_temp': 1.0,
        'traj_tf_similar_user_gate': 'contextual_mlp',
        
        # 这些参数是模型需要的，但在Optuna中没有调整
        'v_orig_dim_seq': TRANSE_EMBED_DIM,
        'c_orig_dim_seq': TRANSE_EMBED_DIM,
        'v_orig_dim_traj': TRANSE_EMBED_DIM,
        'c_orig_dim_traj': TRANSE_EMBED_DIM,
    }
    
    # 我们为每次运行设置一个固定的epoch数，这个数要足够看到趋势，但又不能太长
    # 导致整个分析过程耗时过久。
    EPOCHS_PER_RUN = 15
    print(f"每个配置将训练 {EPOCHS_PER_RUN} 个 epochs。")
    
    # 这个字典将存储我们的结果
    results = {
        'embedding_dim_d': [],
        'num_similar_users_K': [],
        'contrastive_loss_beta': []
    }

    # ==============================================================================
    # 2. 定义分析任务
    # ==============================================================================
    analysis_tasks = [
        {
            'name': '嵌入维度 (d)',
            'param_key': 'embedding_dim_d',
            'values': [32, 64, 128, 256]
        },
        {
            'name': '相似用户数量 (K)',
            'param_key': 'num_similar_users_K',
            'values': [5, 10, 15, 20]
        },
        {
            'name': '对比学习损失权重 (β)',
            'param_key': 'contrastive_loss_beta',
            'values': [0.05, 0.1, 0.2, 0.5]
        }
    ]

    # ==============================================================================
    # 3. 主分析循环
    # ==============================================================================
    for task in analysis_tasks:
        task_name = task['name']
        param_key = task['param_key']
        param_values = task['values']
        
        print("\n" + f"--- 正在分析: {task_name} ---".center(80, "="))
        
        for value in param_values:
            print(f"\n>>> 正在运行分析: {param_key} = {value}")
            
            # --- 为这次特定的运行设置全新的配置、模型和优化器 ---
            current_config = BASE_CONFIG.copy()
            
            # 根据正在测试的超参数更新配置
            if param_key == 'embedding_dim_d':
                current_config['global_graph_hidden_dim'] = value
                current_config['user_rep_tf_d_model'] = value
                current_config['traj_tf_d_model'] = value
                # 调整注意力头数以保持兼容
                current_config['user_rep_tf_nhead'] = find_best_head_size(value)
                current_config['traj_tf_nhead'] = find_best_head_size(value)
            elif param_key == 'num_similar_users_K':
                current_config['traj_tf_num_similar_k'] = value
            elif param_key == 'contrastive_loss_beta':
                current_config['cl_loss_weight'] = value

            # 实例化模型
            model = POIRecommender(
                config=current_config, **model_init_params,
                poi_embeds_pt=poi_embeds_pt, cat_embeds_pt=cat_embeds_pt
            ).to(DEVICE)

            # 为这个模型实例创建优化器
            optimizer = create_optimizer_with_distinct_param_groups(
                model,
                lr_trial=current_config['learning_rate'],
                user_rep_lr_scale_trial=current_config['user_rep_lr_scale'],
                global_graph_lr_scale=current_config['global_graph_lr_scale'],
                wd_trial=current_config['weight_decay']
            )

            # 创建学习率调度器
            num_training_steps = len(train_loader_g) * EPOCHS_PER_RUN
            scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, num_training_steps)
            
            # 损失函数
            criterion_main = nn.CrossEntropyLoss(label_smoothing=current_config['label_smoothing']).to(DEVICE)
            aux_cat_crit = nn.CrossEntropyLoss(label_smoothing=current_config['label_smoothing'], ignore_index=training_params['cat_pad_idx']).to(DEVICE)
            
            print(f"已为 {param_key} = {value} 初始化模型和优化器。开始训练...")

            # --- 训练循环 ---
            for epoch in range(1, EPOCHS_PER_RUN + 1):
                train_epoch(
                    model, optimizer, scheduler, train_loader_g,
                    user_full_history_dl_aux=None, # 为简化分析，我们暂时不动态更新用户画像
                    criterion_main=criterion_main, aux_cat_crit=aux_cat_crit, aux_venue_crit=None,
                    config=current_config, fixed_params=training_params, device=DEVICE,
                    global_graph_data=global_graph_data
                )
                print(f"  Epoch {epoch}/{EPOCHS_PER_RUN} 已完成。")

            # --- 评估 ---
            print("训练完成。正在验证集上进行评估...")
            _, val_metrics = evaluate(
                model, val_loader_g, criterion_main, DEVICE,
                global_graph_data=global_graph_data, config=current_config
            )
            
            mrr_score = val_metrics.get('MRR', 0.0)
            print(f"  => 结果: {param_key} = {value} 的 MRR = {mrr_score:.5f}")
            
            # 存储结果
            results[param_key].append({'value': value, 'mrr': mrr_score})
            
            # --- 清理 ---
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

    # ==============================================================================
    # 4. 打印并绘制最终结果
    # ==============================================================================
    print("\n" + "="*30 + " 分析完成 " + "="*30)
    print("最终结果:")
    pprint(results)

    # 将结果转换为DataFrame以便于绘图
    results_df_d = pd.DataFrame(results['embedding_dim_d']).rename(columns={'value': 'd', 'mrr': 'MRR'})
    results_df_K = pd.DataFrame(results['num_similar_users_K']).rename(columns={'value': 'K', 'mrr': 'MRR'})
    results_df_beta = pd.DataFrame(results['contrastive_loss_beta']).rename(columns={'value': 'β', 'mrr': 'MRR'})

    # 绘图
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16)

    # 绘制 d 的图
    sns.lineplot(ax=axes[0], data=results_df_d, x='d', y='MRR', marker='o')
    axes[0].set_title('(a) Effect of Embedding Dimension $d$')
    axes[0].set_xlabel('Embedding Dimension, $d$')
    axes[0].set_ylabel('MRR')

    # 绘制 K 的图
    sns.lineplot(ax=axes[1], data=results_df_K, x='K', y='MRR', marker='o')
    axes[1].set_title('(b) Effect of Number of Similar Users $K$')
    axes[1].set_xlabel('Number of Similar Users, $K$')
    axes[1].set_ylabel('MRR')

    # 绘制 β 的图
    sns.lineplot(ax=axes[2], data=results_df_beta, x='β', y='MRR', marker='o')
    axes[2].set_title('(c) Effect of Contrastive Loss Weight $β$')
    axes[2].set_xlabel('Contrastive Loss Weight, $β$')
    axes[2].set_ylabel('MRR')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图像
    save_path = os.path.join(MODEL_SAVE_DIR, 'hyperparameter_sensitivity_cn.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n图表已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    set_seed(SEED)
    print(f"使用设备: {DEVICE}")

    df_proc = load_and_preprocess_data(DATA_PATH)
    global_distance_lookup = load_precomputed_distances(DISTANCE_DATA_PATH)
    
    train_ids, val_ids, test_ids = split_users(df_proc, VAL_USER_RATIO, TEST_USER_RATIO)
    global_train_user_ids_list_for_rep_calc = list(train_ids)
    df_tr = df_proc[df_proc['user_id'].isin(train_ids)]
    df_v = df_proc[df_proc['user_id'].isin(val_ids)]
    grp_tr_main = df_tr.groupby('user_id')
    grp_v_main = df_v.groupby('user_id')

    print("为分析任务创建主任务的训练/验证集 Datasets & DataLoaders...")
    train_ds_main = TrajectoryDataset(grp_tr_main, venue_map, MAX_SEQ_LENGTH, venue_pad_idx, cat_pad_idx, num_categories, is_train=True)
    val_ds_main = TrajectoryDataset(grp_v_main, venue_map, MAX_SEQ_LENGTH, venue_pad_idx, cat_pad_idx, num_categories, is_train=False)
    collate_fn_main = partial(main_task_collate_fn, venue_pad_idx_g=venue_pad_idx, cat_pad_idx_g=cat_pad_idx, time_segment_pad_idx_g=TIME_SEGMENT_PAD_IDX)
    train_loader_main = DataLoader(train_ds_main, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_main, num_workers=4, pin_memory=True, drop_last=True)
    val_loader_main = DataLoader(val_ds_main, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_main, num_workers=4, pin_memory=True)

    poi_counts = df_tr['geo_id'].value_counts()
    sorted_popular_poi_str_ids = poi_counts.index.tolist()
    sorted_popular_poi_indices = [venue_map[poi_id] for poi_id in sorted_popular_poi_str_ids if poi_id in venue_map]
    top_k_poi_indices_for_memory = torch.tensor(sorted_popular_poi_indices, dtype=torch.long)
    
    graph_builder = GlobalPOIGraphBuilder(venue_map, category_map, global_distance_lookup, df_tr)
    global_poi_graph_data = graph_builder.build_global_graph()

    NUM_NEGATIVE_SAMPLES = 64
    category_to_pois = defaultdict(list)
    for i in range(global_poi_graph_data.num_nodes):
        cat_id = int(global_poi_graph_data.x[i, 1].item())
        if cat_id != cat_pad_idx:
            category_to_pois[cat_id].append(i)
    contrastive_samples = []
    all_poi_indices = list(range(global_poi_graph_data.num_nodes))
    for anchor_poi_idx in tqdm(all_poi_indices, desc="生成对比学习样本"):
        anchor_cat_id = int(global_poi_graph_data.x[anchor_poi_idx, 1].item())
        if anchor_cat_id == cat_pad_idx or len(category_to_pois.get(anchor_cat_id, [])) < 2: continue
        positive_candidates = category_to_pois[anchor_cat_id]
        positive_poi_idx = anchor_poi_idx
        while positive_poi_idx == anchor_poi_idx: positive_poi_idx = random.choice(positive_candidates)
        negative_poi_indices = []
        num_negs_added = 0
        while num_negs_added < NUM_NEGATIVE_SAMPLES:
            negative_candidate = random.choice(all_poi_indices)
            candidate_cat_id = int(global_poi_graph_data.x[negative_candidate, 1].item())
            if (negative_candidate != anchor_poi_idx and negative_candidate != positive_poi_idx and candidate_cat_id != anchor_cat_id):
                negative_poi_indices.append(negative_candidate)
                num_negs_added += 1
        contrastive_samples.append({'anchor': anchor_poi_idx, 'positive': positive_poi_idx, 'negatives': negative_poi_indices})

    poi_embeds_pt, cat_embeds_pt = None, None
    if DO_TRANSE_PRETRAINING:
        if os.path.exists(POI_EMBED_PATH) and os.path.exists(CAT_EMBED_PATH):
            print("加载已有的预训练嵌入。")
            poi_embeds_pt = torch.load(POI_EMBED_PATH, map_location=DEVICE)
            cat_embeds_pt = torch.load(CAT_EMBED_PATH, map_location=DEVICE)
        else:
            print("开始TransE预训练...")
            transe_triplets, _, _, num_entities, num_relations = prepare_transe_data(df_tr, user_map, venue_map, category_map)
            pre_trained_entity_embeddings = run_transe_pretraining(transe_triplets, num_entities, num_relations, embed_dim=TRANSE_EMBED_DIM, device=DEVICE, epochs=TRANSE_EPOCHS, batch_size=TRANSE_BATCH_SIZE, lr=TRANSE_LR, margin=TRANSE_MARGIN, neg_ratio=TRANSE_NEG_RATIO)
            _, poi_embeds_pt, cat_embeds_pt = extract_and_save_embeddings(pre_trained_entity_embeddings, user_map, venue_map, category_map, len(user_map), num_venues_with_pad, num_categories_with_pad, TRANSE_EMBED_DIM, USER_EMBED_PATH, POI_EMBED_PATH, CAT_EMBED_PATH)
            poi_embeds_pt, cat_embeds_pt = poi_embeds_pt.to(DEVICE), cat_embeds_pt.to(DEVICE)
            
    # 字典1: 用于模型初始化 (POIRecommender.__init__)
    model_init_params = {
        'num_venues_w_pad': num_venues_with_pad, 
        'venue_pad_idx': venue_pad_idx,
        'num_cats_w_pad': num_categories_with_pad, 
        'cat_pad_idx': cat_pad_idx, 
        'num_categories': num_categories,
        'num_time_segments_w_pad': NUM_TIME_SEGMENTS_W_PAD, 
        'time_segment_pad_idx': TIME_SEGMENT_PAD_IDX,
        'num_pairwise_time_bins': NUM_PAIRWISE_TIME_DIFF_BINS, 
        'max_seq_len': MAX_SEQ_LENGTH,
        'num_edge_time_bins_gat_w_pad': NUM_EDGE_TIME_BINS_W_PAD_GAT, 
        'edge_time_pad_idx_gat': EDGE_TIME_BIN_PAD_IDX_GAT,
        'num_edge_dist_bins_gat_w_pad': NUM_EDGE_DIST_BINS_W_PAD_GAT, 
        'edge_dist_pad_idx_gat': EDGE_DIST_BIN_PAD_IDX_GAT,
        'top_k_poi_indices_for_memory': top_k_poi_indices_for_memory
    }
    
    # 字典2: 用于训练过程 (train_epoch)
    training_params = {
        'contrastive_samples': contrastive_samples, 
        'num_cl_negatives': NUM_NEGATIVE_SAMPLES,
        # 把pad_idx也放进来，方便train_epoch函数访问
        'cat_pad_idx': cat_pad_idx,
        'venue_pad_idx': venue_pad_idx,
        'num_cats_w_pad': num_categories_with_pad,
    }

    run_sensitivity_analysis(
        train_loader_g=train_loader_main,
        val_loader_g=val_loader_main,
        model_init_params=model_init_params,
        training_params=training_params,
        poi_embeds_pt=poi_embeds_pt,
        cat_embeds_pt=cat_embeds_pt,
        global_graph_data=global_poi_graph_data
    )
