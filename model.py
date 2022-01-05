import torch
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig


class ClassifyNet(torch.nn.Module):
    def __init__(self, label_number, feature_layers, bert_hidden_size):
        super(ClassifyNet, self).__init__()

        self.feature_layers = feature_layers
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(feature_layers * bert_hidden_size, label_number)

    def forward(self, x):
        out = x['hidden_states']
        out = torch.cat([out[-i][:, 0] for i in range(1, self.feature_layers + 1)], dim=-1)
        out = self.dropout(out)
        out = self.linear(out)

        return out

class BertWrapper(torch.nn.Module):
    def __init__(self, bert: BertModel):
        super(BertWrapper, self).__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids):
        if len(input_ids) == 1:
            return self.bert(input_ids=input_ids[0],
                             attention_mask=attention_mask[0],
                             token_type_ids=token_type_ids[0],
                             output_hidden_states=True)
        else:
            return self.bert(input_ids=input_ids[0],
                             attention_mask=attention_mask[0],
                             token_type_ids=token_type_ids[0],
                             output_hidden_states=True), \
                   self.bert(input_ids=input_ids[1],
                             attention_mask=attention_mask[1],
                             token_type_ids=token_type_ids[1],
                             output_hidden_states=True)

    def get_bert(self):
        return self.bert
