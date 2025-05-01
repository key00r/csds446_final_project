import torch
from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from code.ResultSaving import ResultSaving
from code.Settings import Settings
from code.EvaluateAcc import EvaluateAcc

dataset_name = 'cora'
k = 7

# 加载数据
data_obj = DatasetLoader()
data_obj.dataset_source_folder_path = '././data/' + dataset_name + '/'
data_obj.dataset_name = dataset_name
data_obj.k = k
data_obj.load_all_tag = True

# 配置（必须和fine-tuning阶段完全一致）
bert_config = GraphBertConfig(
    residual_type='graph_raw', k=k, x_size=1433, y_size=7, hidden_size=32,
    intermediate_size=32, num_attention_heads=2, num_hidden_layers=2
)

# 加载fine-tuned模型
model_path = "D:\programs\pycharm\programs\CSDS446MLG\project\Graph-Bert-master\\result\GraphBert\cora_graphbert_finetuning.pth"
method_obj = torch.load(model_path)
method_obj.eval()

# Result 和 Evaluate（评测时不保存结果，因此可以不具体配置）
result_obj = ResultSaving()
evaluate_obj = EvaluateAcc('', '')

# 使用Setting自动准备数据到模型
setting_obj = Settings()
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

# 开始评测
with torch.no_grad():
    output = method_obj.forward(
        method_obj.data['raw_embeddings'],
        method_obj.data['wl_embedding'],
        method_obj.data['int_embeddings'],
        method_obj.data['hop_embeddings'],
        method_obj.data['idx_test']
    )

evaluate_obj.data = {
    'true_y': method_obj.data['y'][method_obj.data['idx_test']],
    'pred_y': output.max(1)[1]
}
acc_test = evaluate_obj.evaluate()

print(f'Fine-tuned Graph-Bert Test Accuracy: {acc_test:.4f}')
