#---- Task 3: Graph-BERT Link Prediction (Cora, Real DatasetLoader) ----
if 1:
    from code.DatasetLoader import DatasetLoader
    from code.MethodBertComp import GraphBertConfig
    from code.MethodGraphBertGraphRecovery import MethodGraphBertGraphRecovery
    from sklearn.metrics import accuracy_score
    import torch
    import random

    dataset_name = 'cora'
    print('************ Start Link Prediction (Structure Recovery) ************')

    # ---- 图结构常量 ----
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

    # ---- 数据加载 ----
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = 7
    data_obj.load_all_tag = True
    data = data_obj.load()

    edges = data['edges']  # ndarray
    raw_embeddings = data['raw_embeddings']
    wl_embedding = data['wl_embedding']
    hop_embeddings = data['hop_embeddings']
    int_embeddings = data['int_embeddings']

    # ---- 构造正负样本 ----
    #edges = [(int(u), int(v)) for u, v in edges if u < v]
    edges = [(int(u), int(v)) for u, v in edges if u < ngraph and v < ngraph and u != v]
    edges = [(u, v) for u, v in edges if u < v]

    test_size = min(1000, int(1 * len(edges)))
    test_pos_edges = random.sample(edges, test_size)

    all_possible = set((i, j) for i in range(ngraph) for j in range(i + 1, ngraph))
    edge_set = set(edges)
    test_neg_edges = random.sample(list(all_possible - edge_set), test_size)

    test_edges = test_pos_edges + test_neg_edges
    test_labels = [1] * test_size + [0] * test_size

    print(f"Test edges: {len(test_edges)}, Pos: {test_size}, Neg: {test_size}")

    # ---- 模型初始化 ----
    bert_config = GraphBertConfig(
        residual_type='graph_raw',
        k=7,
        x_size=nfeature,
        y_size=nclass,
        hidden_size=32,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=2
    )

    method_obj = MethodGraphBertGraphRecovery(bert_config)
    method_obj.eval()
    pretrained_path = './result/GraphBert/cora_graphbert_pretrained_structurerecovery.pth'
    state_dict = torch.load(pretrained_path, weights_only=False).state_dict()
    method_obj.load_state_dict(state_dict, strict=False)

    print("✅ Pretrained structure recovery model loaded.")

    # ---- 前向传播得到节点相似度矩阵 ----
    with torch.no_grad():
        sim_matrix = method_obj.forward(raw_embeddings, wl_embedding, int_embeddings, hop_embeddings)  # [N, N]

    # ---- 执行预测 ----
    preds = []
    for (u, v) in test_edges:
        score = torch.sigmoid(sim_matrix[u][v]).item()
        pred = 1 if score >= 0.5 else 0
        preds.append(pred)

    acc = accuracy_score(test_labels, preds)
    print(f"🔍 Link Prediction Accuracy: {acc:.4f}")
    print('************ Finish Link Prediction ************')


from sklearn.metrics import roc_auc_score, average_precision_score

auc = roc_auc_score(test_labels, [torch.sigmoid(sim_matrix[u][v]).item() for (u, v) in test_edges])
ap = average_precision_score(test_labels, [torch.sigmoid(sim_matrix[u][v]).item() for (u, v) in test_edges])

print(f"🔍 AUC: {auc:.4f}")
print(f"🔍 Average Precision (AP): {ap:.4f}")
