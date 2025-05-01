import torch

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert

import time

from sklearn.cluster import KMeans

BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertGraphClustering(BertPreTrainedModel):
    learning_record_dict = {}
    use_raw_feature = True
    cluster_number = 0
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config):
        super(MethodGraphBertGraphClustering, self).__init__(config)
        self.config = config
        self.bert = MethodGraphBert(config)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids):

        outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        kmeans = KMeans(n_clusters=self.cluster_number, max_iter=self.max_epoch)
        if self.use_raw_feature:
            clustering_result = kmeans.fit_predict(self.data['X'])
        else:
            clustering_result = kmeans.fit_predict(sequence_output.tolist())

        return {'pred_y': clustering_result, 'true_y': self.data['y']}

    def train_model(self, max_epoch):
        # t_begin = time.time()
        #
        # clustering = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'],
        #                       self.data['hop_embeddings'])
        #
        # self.learning_record_dict = clustering
        import numpy as np
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
        from scipy.optimize import linear_sum_assignment

        def clustering_acc(y_true, y_pred):
            D = max(y_pred.max(), y_true.max()) + 1
            w = np.zeros((D, D), dtype=np.int64)
            for i in range(y_pred.size):
                w[y_pred[i], y_true[i]] += 1
            row_ind, col_ind = linear_sum_assignment(w.max() - w)
            return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / y_pred.size

        t_begin = time.time()

        clustering = self.forward(
            self.data['raw_embeddings'],
            self.data['wl_embedding'],
            self.data['int_embeddings'],
            self.data['hop_embeddings']
        )

        pred_y = np.array(clustering['pred_y'])
        true_y = self.data['y'].detach().cpu().numpy()

        acc = clustering_acc(true_y, pred_y)
        nmi = normalized_mutual_info_score(true_y, pred_y)
        ari = adjusted_rand_score(true_y, pred_y)

        self.learning_record_dict = {
            'acc': acc,
            'nmi': nmi,
            'ari': ari,
            'time': time.time() - t_begin
        }

    def run(self):

        self.train_model(self.max_epoch)
        print("🔍 Clustering Result:")
        for k, v in self.learning_record_dict.items():
            if isinstance(v, float):
                print(f"{k.upper():>8}: {v:.4f}")
            else:
                print(f"{k.upper()}: {v}")

        return self.learning_record_dict