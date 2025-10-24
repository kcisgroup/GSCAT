import os
import pandas as pd

def analyze_checkin_data(file_path, dataset_name):
    """
    读取单个签到数据CSV文件，并计算详细的统计信息。

    Args:
        file_path (str): CSV文件的路径。
        dataset_name (str): 用于在结果中标示该数据集的名称。

    Returns:
        dict: 包含所有统计信息的字典。
    """
    print(f"正在从 '{file_path}' ({dataset_name}) 加载数据...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"({dataset_name}) 数据加载完成，开始计算统计信息...")
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return None

    # 计算各项统计数据
    num_users = df['user_id'].nunique()
    num_pois = df['geo_id'].nunique()
    num_categories = df['venue_category_id'].nunique()
    num_checkins = len(df)
    avg_checkins_per_user = num_checkins / num_users if num_users > 0 else 0
    
    if num_users > 0 and num_pois > 0:
        sparsity = 1.0 - (num_checkins / (num_users * num_pois))
    else:
        sparsity = 1.0

    # 将统计结果存入字典
    stats = {
        'Dataset': dataset_name,
        '#Users': num_users,
        '#POIs': num_pois,
        '#Categories': num_categories,
        '#Check-ins': num_checkins,
        'Avg. Check-ins/User': f"{avg_checkins_per_user:.2f}",
        'Sparsity': f"{sparsity:.6f}"
    }
    
    return stats

def analyze_and_compare_datasets(dataset_info):
    """
    分析多个数据集并将它们的统计信息进行对比。

    Args:
        dataset_info (dict): 一个字典，键是数据集名称，值是文件路径。
                             例如: {'NYC': 'nyc.csv', 'TKY': 'tky.csv'}

    Returns:
        pd.DataFrame: 包含所有数据集对比统计信息的数据框。
    """
    all_stats = []
    
    for name, path in dataset_info.items():
        stats = analyze_checkin_data(path, name)
        if stats:
            all_stats.append(stats)
            
    if not all_stats:
        print("未能成功分析任何数据集。")
        return None
        
    # 将字典列表转换为DataFrame
    comparison_df = pd.DataFrame(all_stats)
    # 设置 'Dataset' 列为索引，使表格更清晰
    comparison_df.set_index('Dataset', inplace=True)
    
    return comparison_df

def print_comparison_table(comparison_df):
    """
    以美观的表格形式打印数据集的对比统计数据。
    """
    if comparison_df is None or comparison_df.empty:
        return
        
    print("\n" + "="*85)
    print(" " * 28 + "数据集统计信息对比摘要")
    print("="*85)
    
    # 直接打印DataFrame，pandas会自动格式化
    print(comparison_df)
    
    print("="*85)
    print("\n说明:")
    print("- #Users: 数据集中独立用户的总数。")
    print("- #POIs: 数据集中独立兴趣点(POI)的总数。")
    print("- #Categories: 数据集中独立POI类别的总数。")
    print("- #Check-ins: 数据集中签到记录的总数。")
    print("- Avg. Check-ins/User: 平均每个用户的签到次数。")
    print("- Sparsity: 用户-POI交互矩阵的稀疏度。值越接近1.0，数据越稀疏。")
    print("-" * 85)



BASE_DIR = os.getcwd()
NYC__PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_nyc_cleaned.csv')
TKY__PATH = os.path.join(BASE_DIR, 'MyModel/dataset_process/foursquare_tky_cleaned.csv')

if __name__ == "__main__":

    datasets_to_analyze = {
        'Foursquare-NYC': NYC__PATH,
        'Foursquare-TKY': TKY__PATH
    }
    # 执行分析并打印结果
    comparison_dataframe  = analyze_and_compare_datasets(datasets_to_analyze)
    print_comparison_table(comparison_dataframe)