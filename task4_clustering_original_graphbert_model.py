#---- Task 4: Graph-BERT Clustering via MethodGraphBertGraphClustering + pretrained encoder ----
if 1:
    from code.DatasetLoader import DatasetLoader
    from code.MethodBertComp import GraphBertConfig
    from code.MethodGraphBertGraphClustering import MethodGraphBertGraphClustering
    from code.ResultSaving import ResultSaving
    from code.Settings import Settings
    import torch

    dataset_name = 'cora'

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

    print('************ Start Clustering via GraphBert ************')

    # ---- Hyperparameters ----
    k = 7
    max_epoch = 1
    lr = 0.001
    residual_type = 'graph_raw'
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    x_size = nfeature
    y_size = nclass
    graph_size = ngraph

    # ---- Load data ----
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    # ---- Initialize model ----
    bert_config = GraphBertConfig(
        residual_type=residual_type,
        k=k,
        x_size=nfeature,
        y_size=y_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers
    )
    method_obj = MethodGraphBertGraphClustering(bert_config)
    method_obj.cluster_number = y_size
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr

    # ---- Load pretrained encoder (structure recovery) ----
    pretrained_path = './result/GraphBert/cora_graphbert_pretrained_structurerecovery.pth'
    pretrained_model = torch.load(pretrained_path, weights_only=False)
    pretrained_dict = pretrained_model.state_dict()
    model_dict = method_obj.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    method_obj.load_state_dict(model_dict)
    print(f"✅ Loaded pretrained encoder from: {pretrained_path}")

    # ---- Setup setting and result ----
    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/GraphBert/clustering_' + dataset_name
    result_obj.result_destination_file_name = 'graphbert_clustering_from_pretrained_encoder'

    setting_obj = Settings()
    evaluate_obj = None

    # ---- Run clustering ----
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    record = setting_obj.load_run_save_evaluate()

    if isinstance(record, dict):
        print("🔍 Clustering Evaluation Results:")
        for k, v in record.items():
            print(f"{k.upper():>8}: {v:.4f}" if isinstance(v, float) else f"{k.upper()}: {v}")

    print('************ Finish Clustering ************')


