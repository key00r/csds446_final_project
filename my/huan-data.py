import os
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import pickle
import gzip
import urllib.request
from tqdm import tqdm

# 配置路径
DATASET_URL = "https://snap.stanford.edu/data/ca-GrQc.txt.gz"
DATA_DIR = "./data/cora/"
OUTPUT_NODE = os.path.join(DATA_DIR, "node")
OUTPUT_LINK = os.path.join(DATA_DIR, "link")
TEMP_GRQC = os.path.join(DATA_DIR, "ca-GrQc.txt.gz")
EXTRACTED_GRQC = os.path.join(DATA_DIR, "ca-GrQc.txt")


def download_file(url, local_path):
    """下载文件到本地路径"""
    if os.path.exists(local_path):
        print(f"文件已存在: {local_path}")
        return

    print(f"正在下载: {url} -> {local_path}")
    urllib.request.urlretrieve(url, local_path)
    print("下载完成!")


def extract_gz(gz_path, output_path):
    """解压.gz文件"""
    if os.path.exists(output_path):
        print(f"解压文件已存在: {output_path}")
        return

    print(f"正在解压: {gz_path} -> {output_path}")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    print("解压完成!")


def preprocess_ca_grqc():
    """
    预处理ca-GrQc数据集，创建node和link文件
    """
    print("开始预处理ca-GrQc数据集...")

    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)

    # 下载和解压数据集（如果需要）
    if not os.path.exists(EXTRACTED_GRQC):
        if not os.path.exists(TEMP_GRQC):
            download_file(DATASET_URL, TEMP_GRQC)
        extract_gz(TEMP_GRQC, EXTRACTED_GRQC)

    # 从边列表创建NetworkX图
    G = nx.Graph()

    # 读取边列表，跳过注释行
    print("正在加载图数据...")
    with open(EXTRACTED_GRQC, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                src, dst = int(parts[0]), int(parts[1])
                G.add_edge(src, dst)

    print(f"原始图统计: {len(G.nodes())}个节点, {len(G.edges())}条边")

    # 获取所有节点并重新标记为从0开始的连续整数
    all_nodes = sorted(G.nodes())
    node_map = {node: i for i, node in enumerate(all_nodes)}
    G = nx.relabel_nodes(G, node_map)

    print("计算节点特征...")
    # 计算节点特征：使用度、中心性指标和聚类系数
    degrees = dict(G.degree())

    print("计算介数中心性（可能需要较长时间）...")
    # 使用采样以加快介数中心性计算，最多采样500个节点
    betweenness = nx.betweenness_centrality(G, k=min(500, len(G.nodes)))

    print("计算聚类系数...")
    clustering = nx.clustering(G)

    # 计算pagerank
    pagerank = nx.pagerank(G, alpha=0.85)

    # 组合特征
    features = []
    for node in range(len(G.nodes())):
        node_features = [
            degrees.get(node, 0),
            betweenness.get(node, 0),
            clustering.get(node, 0),
            pagerank.get(node, 0)
        ]
        features.append(node_features)

    # 标准化特征
    features = np.array(features, dtype=np.float32)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 扩展特征维度，匹配cora的1433维
    print("生成扩展特征 (1433维)...")
    expanded_features = np.zeros((len(G.nodes()), 1433))

    # 将计算的特征放入前4列
    expanded_features[:, :features.shape[1]] = features

    # 使用度信息生成一些随机特征填充剩余列
    np.random.seed(42)  # 确保可重现性
    for i in range(len(G.nodes())):
        deg = degrees.get(i, 1)
        node_seed = i * 42  # 为每个节点使用不同的种子
        np.random.seed(node_seed)
        expanded_features[i, features.shape[1]:] = np.random.normal(
            scale=0.01 * deg, size=1433 - features.shape[1]
        )

    # 使用社区检测为节点分配类别标签（7类，与cora相同）
    print("进行社区检测以分配类别标签...")
    try:
        # 尝试使用greedy_modularity_communities
        communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    except Exception as e:
        print(f"社区检测失败: {e}，将使用连通分量作为替代")
        communities = list(nx.connected_components(G))

    print(f"检测到{len(communities)}个社区")

    # 将每个节点映射到其社区（限制为7个类别，与cora相同）
    node_to_community = {}
    for i, community in enumerate(communities):
        community_id = min(i, 6)  # 限制为7个类别 (0-6)
        for node in community:
            node_to_community[node] = community_id

    # 创建node文件（ID，特征，标签）
    print(f"创建node文件: {OUTPUT_NODE}")
    with open(OUTPUT_NODE, 'w') as f:
        for node in range(len(G.nodes())):
            # 获取节点特征
            feature_str = ' '.join([str(val) for val in expanded_features[node]])
            # 获取社区（类别）ID，默认为0
            class_id = node_to_community.get(node, 0)
            # 写入文件：ID，特征，类别
            f.write(f"{node} {feature_str} {class_id}\n")

    # 创建link文件
    print(f"创建link文件: {OUTPUT_LINK}")
    with open(OUTPUT_LINK, 'w') as f:
        for src, dst in G.edges():
            f.write(f"{src} {dst}\n")

    print("预处理完成！")
    print(f"节点文件已保存到: {OUTPUT_NODE}")
    print(f"链接文件已保存到: {OUTPUT_LINK}")
    print(f"节点数: {len(G.nodes())}, 边数: {len(G.edges())}")
    print("这些文件可以直接替换原始cora数据集的文件。")

    # 删除临时文件
    if os.path.exists(TEMP_GRQC):
        os.remove(TEMP_GRQC)
    if os.path.exists(EXTRACTED_GRQC):
        os.remove(EXTRACTED_GRQC)

    return True


if __name__ == "__main__":
    preprocess_ca_grqc()