from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from code.MethodGraphBertGraphClustering import MethodGraphBertGraphClustering
from code.ResultSaving import ResultSaving
from code.Settings import Settings
import numpy as np
import torch


#---- 'cora' , 'citeseer', 'pubmed' ----

dataset_name = 'cora'

np.random.seed(42)
torch.manual_seed(12)

#---- cora-small is for debuging only ----
if dataset_name == 'cora-small':
    nclass = 7
    nfeature = 1433
    ngraph = 10
elif dataset_name == 'cora':
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


#---- Fine-Tuning Task 1: Graph Bert Node Classification (Cora, Citeseer, and Pubmed) ----
if 1:
    #---- hyper-parameters ----
    if dataset_name == 'pubmed':
        lr = 0.001
        k = 30
        max_epoch = 1000 # 500 ---- do an early stop when necessary ----
    elif dataset_name == 'cora':
        lr = 0.01
        k = 7
        max_epoch = 800 # 150 ---- do an early stop when necessary ----
    elif dataset_name == 'citeseer':
        k = 5
        lr = 0.001
        max_epoch = 2000 #2000 # it takes a long epochs to get good results, sometimes can be more than 2000

    x_size = nfeature
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'
    # --------------------------

    print('************ Start ************')
    print('GrapBert, dataset: ' + dataset_name + ', residual: ' + residual_type + ', k: ' + str(k) + ', hidden dimension: ' + str(hidden_size) +', hidden layer: ' + str(num_hidden_layers) + ', attention head: ' + str(num_attention_heads))
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
    #method_obj = MethodGraphBertNodeClassification(bert_config)
    # pretrained_model = torch.load('./result/GraphBert/cora_graphbert_pretrained.pth', weights_only=False)
    # method_obj = MethodGraphBertNodeClassification(bert_config)
    # method_obj.load_state_dict(pretrained_model.state_dict(), strict=False)

    # 1. 初始化分类模型
    method_obj = MethodGraphBertNodeClassification(bert_config)

    # 2. 加载预训练模型（原始模型构造的是 node reconstruction）
    pretrained_model = torch.load('./result/GraphBert/cora_graphbert_pretrained.pth', weights_only=False)

    # 3. 只加载它的 bert 编码器部分（参数名以 'bert.' 开头）
    pretrained_dict = pretrained_model.state_dict()
    model_dict = method_obj.state_dict()

    # 4. 过滤掉那些不匹配的参数（比如 cls_y）
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

    # 5. 更新当前模型的参数
    model_dict.update(filtered_dict)
    method_obj.load_state_dict(model_dict)

    print(f"✅ Loaded {len(filtered_dict)} parameters from pretrained model.")

    # 加载预训练参数（不是整个模型）
    # state_dict = torch.load('./result/GraphBert/cora_graphbert_pretrained.pth', weights_only=False)
    # method_obj.load_state_dict(state_dict, strict=False)  # 注意 strict=False

    # 新增以下代码，用于加载预训练模型
    # # 直接加载整个模型
    # pretrained_path = './result/GraphBert/cora_7_node_reconstruction'
    # method_obj = torch.load(pretrained_path, weights_only=False)
    # method_obj.eval()
    # 修改为直接加载完整的预训练模型
    # pretrained_model_path = './result/GraphBert/cora_graphbert_pretrained.pth'
    # method_obj = torch.load(pretrained_model_path, weights_only=False)


    #---- set to false to run faster ----
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/GraphBert/'
    result_obj.result_destination_file_name = dataset_name + '_' + str(num_hidden_layers)

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------
    # 增加明确的模型保存 (最新PyTorch推荐)
    torch.save(method_obj, './result/GraphBert/' + dataset_name + '_graphbert_finetuning_rope.pth')

    method_obj.save_pretrained('./result/PreTrained_GraphBert/' + dataset_name + '/node_classification_complete_model/')
    print('************ Finish ************')
#------------------------------------


#---- Fine-Tuning Task 2: Graph Bert Graph Clustering (Cora, Citeseer, and Pubmed) ----
if 0:
    #---- hyper-parameters ----
    if dataset_name == 'pubmed':
        lr = 0.001
        k = 30
        max_epoch = 500 # 500 ---- do an early stop when necessary ----
    elif dataset_name == 'cora':
        lr = 0.01
        k = 7
        max_epoch = 150 #150 # ---- do an early stop when necessary ----
    elif dataset_name == 'citeseer':
        k = 5
        lr = 0.001
        max_epoch = 2000 #2000 # it takes a long epochs to converge, probably more than 2000

    x_size = nfeature
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'
    # --------------------------

    print('************ Start ************')
    print('GrapBert, dataset: ' + dataset_name + ', residual: ' + residual_type + ', k: ' + str(k) + ', hidden dimension: ' + str(hidden_size) +', hidden layer: ' + str(num_hidden_layers) + ', attention head: ' + str(num_attention_heads))
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True

    bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=y_size, hidden_size=hidden_size, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
    method_obj = MethodGraphBertGraphClustering(bert_config)
    #---- set to false to run faster ----
    method_obj.cluster_number = y_size
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/GraphBert/clustering_' + dataset_name
    result_obj.result_destination_file_name = '_' + ''

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    print('************ Finish ************')
#------------------------------------
