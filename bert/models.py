import torch.nn as nn
import torch 
import transformers


class BERTBasedUncased(nn.Module):
    def __init__(self, bert_path):
        super(BERTBasedUncased,self).__init__()

        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path) # Load the model from pretrained
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768,30) # Getting 30 dimension ouput as result

    def forward(self,ids, mask, token_type_ids):
        _, output2 = self.bert(ids, attention_mask = mask,token_type_ids=token_type_ids) # Bert have 2 output and the second one is our need
        finaloupt = self.out(output2)
        return finaloupt