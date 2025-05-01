import torch
from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from code.ResultSaving import ResultSaving
from code.Settings import Settings
from code.EvaluateAcc import EvaluateAcc

if 1:
    dataset_name = 'cora'
    k = 7

    # 加载数据
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    # 配置（必须和fine-tuning阶段完全一致）
    bert_config = GraphBertConfig(
        residual_type='graph_raw', k=k, x_size=1433, y_size=7, hidden_size=32,
        intermediate_size=32, num_attention_heads=2, num_hidden_layers=2
    )

    # 加载fine-tuned模型
    model_path = "D:\programs\pycharm\programs\CSDS446MLG\project\Graph-Bert-master\\result\GraphBert\cora_graphbert_finetuning_rope.pth"
    method_obj = torch.load(model_path,weights_only=False)
    method_obj.eval()

    # Result 和 Evaluate（评测时不保存结果，因此可以不具体配置）
    result_obj = ResultSaving()
    evaluate_obj = EvaluateAcc('', '')

    # 使用Setting自动准备数据到模型
    setting_obj = Settings()
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

    # 开始评测
    with torch.no_grad():
        idx_test = method_obj.data['idx_test']

        output = method_obj.forward(
            method_obj.data['raw_embeddings'],
            method_obj.data['wl_embedding'],
            method_obj.data['int_embeddings'],
            method_obj.data['hop_embeddings'],
            idx=idx_test  # 注意这点！
        )
        # print(output.shape)
        # exit()

    evaluate_obj.data = {
            'true_y': method_obj.data['y'][idx_test],
            'pred_y': output.max(1)[1]
        }

    acc_test = evaluate_obj.evaluate()

    # 2. 手动截取测试集的预测结果
    # idx_test = method_obj.data['idx_test']
    # outputtest = output[idx_test]  # shape = [1000, num_classes]
    #
    # # 3. 评估
    # evaluate_obj.data = {
    #     'true_y': method_obj.data['y'][idx_test],
    #     'pred_y': outputtest.max(1)[1]
    # }
    # acc_test = evaluate_obj.evaluate()

    print(f'Fine-tuned Graph-Bert Test Accuracy: {acc_test:.4f}')

# ---- Task 3: Graph-BERT Link Prediction (Cora, using pretrained Structure Recovery model) ----
if 0:
    from code.DatasetLoader import DatasetLoader
    from code.MethodBertComp import GraphBertConfig
    from code.MethodGraphBertGraphRecovery import MethodGraphBertGraphRecovery
    from sklearn.metrics import accuracy_score
    import numpy as np
    import torch
    import random


    dataset_name = 'cora'
    # 初始化图数据维度
    if dataset_name == 'cora':
        nclass = 7
        nfeature = 1433
        ngraph = 2708
    elif dataset_name == 'citeseer':
        nclass = 6
        nfeature = 3703
        ngraph = 3312
    elif dataset_name == 'pubmed':
        nclass = 3
        nfeature = 500
        ngraph = 19717

    print('************ Start Link Prediction (Structure Recovery) ************')

    # ---- 参数设置 ----
    lr = 0.001
    k = 7
    max_epoch = 1  # 不训练，仅测试结构恢复效果

    residual_type = 'graph_raw'
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    x_size = nfeature
    y_size = nclass

    # ---- 数据加载 ----
    dataset_name = 'cora'
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    # 构造原始边集合
    #adj_matrix = data_obj.adj.to_dense()
    # 从 edge_index 构造邻接矩阵（稀疏 -> 稠密）
    edge_index = torch.LongTensor(data_obj.edge_index)

    adj_matrix = torch.zeros((ngraph, ngraph))
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1  # 如果是无向图

    edges = adj_matrix.nonzero(as_tuple=False).tolist()
    edges = [(u, v) for u, v in edges if u < v]  # 去掉重复边

    num_nodes = adj_matrix.size(0)
    num_edges = len(edges)
    print(f"Total edges in graph: {num_edges}")

    # ---- 构造测试集（正例 + 负例） ----
    test_size = min(1000, int(0.2 * num_edges))
    test_pos_edges = random.sample(edges, test_size)

    # 构造相同数量的负例（随机节点对，但不在原图中有边）
    all_possible = set((i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes))
    edge_set = set(edges)
    test_neg_edges = random.sample(list(all_possible - edge_set), test_size)

    test_edges = test_pos_edges + test_neg_edges
    test_labels = [1] * test_size + [0] * test_size

    print(f"Test edges: {len(test_edges)}, Pos: {test_size}, Neg: {test_size}")

    # ---- 模型初始化 ----
    bert_config = GraphBertConfig(
        residual_type=residual_type,
        k=k,
        x_size=x_size,
        y_size=y_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers
    )

    method_obj = MethodGraphBertGraphRecovery(bert_config)
    method_obj.lr = lr
    method_obj.max_epoch = max_epoch

    # ✅ 加载预训练结构恢复模型
    pretrained_model = torch.load('./result/GraphBert/cora_graphbert_pretrained_structurerecovery.pth', weights_only=False)
    method_obj.load_state_dict(pretrained_model.state_dict(), strict=False)

    method_obj.eval()
    print("✅ Loaded pretrained structure recovery model.")

    # ---- 准备输入数据 ----
    raw_embeddings = torch.FloatTensor(data_obj.x)
    wl_embedding = torch.LongTensor(data_obj.node_wl_embedding)
    int_embeddings = torch.LongTensor(data_obj.intimacy_embedding)
    hop_embeddings = torch.LongTensor(data_obj.hop_embedding)

    # ---- 预测函数（根据 attention 或 similarity） ----
    def predict_link_score(method, u, v):
        with torch.no_grad():
            z = method.forward(raw_embeddings, wl_embedding, int_embeddings, hop_embeddings)  # [N, D]
            # 相似度（或 attention-based 重构）
            score = torch.sigmoid((z[u] * z[v]).sum()).item()
        return score

    # ---- 执行测试集预测 ----
    preds = []
    for (u, v) in test_edges:
        score = predict_link_score(method_obj, u, v)
        pred = 1 if score >= 0.5 else 0
        preds.append(pred)

    acc = accuracy_score(test_labels, preds)
    print(f"🔍 Link Prediction Accuracy: {acc:.4f}")
    print('************ Finish Link Prediction ************')
