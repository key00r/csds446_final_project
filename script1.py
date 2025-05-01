import numpy as np
import torch
import os
import networkx as nx
import pickle

from code.DatasetLoader import DatasetLoader
from code.MethodWLNodeColoring import MethodWLNodeColoring
from code.MethodGraphBatching import MethodGraphBatching
from code.MethodHopDistance import MethodHopDistance
from code.ResultSaving import ResultSaving
from code.Settings import Settings

# ---- 'cora' , 'citeseer', 'pubmed' ----
dataset_name = 'cora'  # 保持名称不变

np.random.seed(42)
torch.manual_seed(12)

# 创建必要的目录
os.makedirs('./result/WL/', exist_ok=True)
os.makedirs('./result/Batch/', exist_ok=True)
os.makedirs('./result/Hop/', exist_ok=True)
os.makedirs('./result/RoPE/', exist_ok=True)

# 检测数据集大小
print("检测数据集大小...")
actual_nodes = 0
actual_edges = 0

# 计算节点数
with open(f'./data/{dataset_name}/node', 'r') as f:
    for line in f:
        actual_nodes += 1

# 计算边数
with open(f'./data/{dataset_name}/link', 'r') as f:
    for line in f:
        actual_edges += 1

print(f"检测到数据集: 节点数={actual_nodes}, 边数={actual_edges}")

# 根据实际大小设置参数
if actual_nodes > 5000:  # ca-GrQc约有5242个节点
    print("检测到ca-GrQc数据集")
    nclass = 7  # 保持与cora相同的类别数
    nfeature = 1433  # 保持与cora相同的特征维度
    ngraph = actual_nodes
else:
    # 原始cora参数
    nclass = 7
    nfeature = 1433
    ngraph = 2708

print(f"使用参数: nclass={nclass}, nfeature={nfeature}, ngraph={ngraph}")

# 优化k值范围，对于较大的图可以减少k以提高效率
k_values = [1, 2, 3, 5, 7] if actual_nodes > 5000 else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ---- Step 1: WL based graph coloring ----
if 1:
    print('************ Start ************')
    print('WL, dataset: ' + dataset_name)
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name

    method_obj = MethodWLNodeColoring()
    # 对于大型图，减少WL迭代次数以加快速度
    if actual_nodes > 5000:
        method_obj.max_iter = 1  # 减少WL着色的迭代次数

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/WL/'
    result_obj.result_destination_file_name = dataset_name

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print("初始化WL节点着色...")
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    print("运行WL节点着色...")
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    print('************ Finish ************')
# ------------------------------------

# ---- Step 2: intimacy calculation and subgraph batching ----
if 1:
    for k in k_values:
        print('************ Start ************')
        print('Subgraph Batching, dataset: ' + dataset_name + ', k: ' + str(k))
        # ---- objection initialization setction ---------------
        data_obj = DatasetLoader()
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.dataset_name = dataset_name
        data_obj.compute_s = True

        method_obj = MethodGraphBatching()
        method_obj.k = k

        result_obj = ResultSaving()
        result_obj.result_destination_folder_path = './result/Batch/'
        result_obj.result_destination_file_name = dataset_name + '_' + str(k)

        setting_obj = Settings()

        evaluate_obj = None
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        print(f"初始化子图批处理 k={k}...")
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        print(f"运行子图批处理 k={k}...")
        setting_obj.load_run_save_evaluate()
        # ------------------------------------------------------

        print('************ Finish ************')
# ------------------------------------

# ---- Step 3: Shortest path: hop distance among nodes ----
if 1:
    for k in k_values:
        print('************ Start ************')
        print('HopDistance, dataset: ' + dataset_name + ', k: ' + str(k))
        # ---- objection initialization setction ---------------
        data_obj = DatasetLoader()
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.dataset_name = dataset_name

        method_obj = MethodHopDistance()
        method_obj.k = k
        method_obj.dataset_name = dataset_name

        result_obj = ResultSaving()
        result_obj.result_destination_folder_path = './result/Hop/'
        result_obj.result_destination_file_name = 'hop_' + dataset_name + '_' + str(k)

        setting_obj = Settings()

        evaluate_obj = None
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        print(f"初始化Hop距离计算 k={k}...")
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        print(f"运行Hop距离计算 k={k}...")
        setting_obj.load_run_save_evaluate()
        # ------------------------------------------------------

        print('************ Finish ************')
# ------------------------------------

# ---- Step 4: Compute shortest path distances for RoPE ----
if 1:
    print('************ Start ************')
    print('ShortestPath for RoPE, dataset: ' + dataset_name)
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name

    from code.shortestpath import MethodShortestPath

    method_obj = MethodShortestPath()
    method_obj.dataset_name = dataset_name

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/RoPE/'
    result_obj.result_destination_file_name = 'shortest_paths_' + dataset_name

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print("初始化RoPE最短路径计算...")
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    print("运行RoPE最短路径计算...")

    # 对于大型图，可能需要更多内存和时间，显示进度信息
    if actual_nodes > 5000:
        print(f"注意：处理较大的图({actual_nodes}个节点)，此步骤可能需要较长时间...")

    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    print('************ Finish ************')
# ------------------------------------