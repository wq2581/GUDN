# Reference to LightXML
import torch
from transformers import BertTokenizer, BertConfig, BertModel


class ClassifyNet(torch.nn.Module):
    def __init__(self, feature_layers, bert_hidden_size):
        super(ClassifyNet, self).__init__()
        self.cluster_number=8192
        self.feature_layers = feature_layers
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(feature_layers * bert_hidden_size, self.cluster_number)
        self.l1=torch.nn.Linear(self.cluster_number, 3000)

    def forward(self, x, labels):
        out = x['hidden_states']
        out = torch.cat([out[-i][:, 0] for i in range(1, self.feature_layers + 1)], dim=-1)
        out = self.dropout(out)
        out = self.linear(out)


        l = labels.to(dtype=torch.bool)
        target_candidates = torch.masked_select(candidates, l).detach().cpu()
        target_candidates_num = l.sum(dim=1).detach().cpu()
        logits = out
        if group_gd is not None:
            logits += group_gd
        scores, indices = torch.topk(logits, k=10)
        scores, indices = scores.cpu().detach().numpy(), indices.cpu().detach().numpy()
        candidates, candidates_scores = [], []
        for index, score in zip(indices, scores):
            candidates.append(self.group_y[index])
            candidates_scores.append([np.full(c.shape, s) for c, s in zip(candidates[-1], score)])
            candidates[-1] = np.concatenate(candidates[-1])
            candidates_scores[-1] = np.concatenate(candidates_scores[-1])
        max_candidates = max([i.shape[0] for i in candidates])
        candidates = np.stack([np.pad(i, (0, max_candidates - i.shape[0]), mode='edge') for i in candidates])
        candidates_scores = np.stack([np.pad(i, (0, max_candidates - i.shape[0]), mode='edge') for i in candidates_scores])
        groups, candidates, group_candidates_scores = indices,candidates,candidates_scores
        bs = 0   #start
        new_labels = []
        for i, n in enumerate(target_candidates_num.numpy()):

            be = bs + n    #end
            c = set(target_candidates[bs: be].numpy())

            c2 = candidates[i]

            new_labels.append(torch.tensor([1.0 if i in c else 0.0 for i in c2 ]))
            if len(c) != new_labels[-1].sum():
                s_c2 = set(c2)
                for cc in list(c):
                    if cc in s_c2:
                        continue
                    for j in range(new_labels[-1].shape[0]):
                        if new_labels[-1][j].item() != 1:
                            c2[j] = cc
                            new_labels[-1][j] = 1.0
                            break
            bs = be

        emb = self.l1(logits)
        embed_weights = self.embed(candidates)
        emb = emb.unsqueeze(-1)
        logits = torch.bmm(embed_weights, emb).squeeze(-1)

        return logits



