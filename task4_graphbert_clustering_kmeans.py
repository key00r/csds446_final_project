#---- Task 4: Graph-BERT Clustering (using structure recovery pretrained model) ----
if 1:
    from code.DatasetLoader import DatasetLoader
    from code.MethodBertComp import GraphBertConfig
    from code.MethodGraphBertGraphRecovery import MethodGraphBertGraphRecovery
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    import torch
    import numpy as np

    dataset_name = 'cora'
    print('************ Start Graph Clustering (KMeans) ************')

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

    raw_embeddings = data['raw_embeddings']
    wl_embedding = data['wl_embedding']
    hop_embeddings = data['hop_embeddings']
    int_embeddings = data['int_embeddings']
    labels = data['y']

    # ---- 模型初始化并加载结构恢复模型 ----
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

    print("✅ Pretrained model loaded for clustering.")

    # ---- 提取节点嵌入特征（从 bert 输出中平均得到）----
    with torch.no_grad():
        outputs = method_obj.bert(raw_embeddings, wl_embedding, int_embeddings, hop_embeddings)  # outputs[0] shape: [N, K+1, D]
        k_hop = bert_config.k
        node_representations = sum(outputs[0][:, i, :] for i in range(k_hop + 1)) / (k_hop + 1)  # [N, D]

    embeddings = node_representations.detach().cpu().numpy()
    true_labels = labels.detach().cpu().numpy()

    # ---- KMeans 聚类 ----
    km = KMeans(n_clusters=nclass, random_state=42, n_init=10)
    pred_labels = km.fit_predict(embeddings)
    #
    # def clustering_acc(y_true, y_pred):
    #     from sklearn.utils.linear_assignment_ import linear_assignment
    #     import numpy as np
    #     D = max(y_pred.max(), y_true.max()) + 1
    #     w = np.zeros((D, D), dtype=np.int64)
    #     for i in range(y_pred.size):
    #         w[y_pred[i], y_true[i]] += 1
    #     ind = linear_assignment(w.max() - w)
    #     return sum([w[i, j] for i, j in ind]) / y_pred.size


    def clustering_acc(y_true, y_pred):
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size


    acc = clustering_acc(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    print(f"🔍 Clustering Accuracy (ACC): {acc:.4f}")
    print(f"🔍 Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"🔍 Adjusted Rand Index (ARI): {ari:.4f}")
    print('************ Finish Graph Clustering ************')
