import torch

class BertDatasetTrainning:
    def __init__(self,title, body, answer, targets, tokenizer, max_len):
        self.title = title
        self.body = body
        self.answer = answer # 3type of data we need 

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = targets


    def __len__(self):
        return len(self.title)


    def __getitem__(self,idx):
        title = str(self.title[idx])
        body = str(self.body[idx])
        answer = str(self.answer[idx])

        inputs = self.tokenizer.encode_plus(
            title + ' ' + body,
            answer,
            add_special_tokens = True,
            max_length = self.max_len
        ) # It would return a tokened input  
         
        ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        mask = inputs['attention_mask']

        padding_len = self.max_len - len(ids) # get the padding length

        ids = ids + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        mask = mask  + ([0] * padding_len)

        return {
            "ids" : torch.tensor(ids, dtype = torch.long),
            "mask" : torch.tensor(mask, dtype = torch.long),
            "token_type_ids" : torch.tensor(token_type_ids, dtype = torch.long),
            "targets" : torch.tensor(self.targets[idx,:], dtype = torch.float)

        }