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


class BERTDatasetTest:
    def __init__(self, qtitle, qbody, answer, tokenizer, max_length):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, item):
        question_title = str(self.qtitle[item])
        question_body = str(self.qbody[item])
        answer_text = str(self.answer[item])

        question_title = " ".join(question_title.split())
        question_body = " ".join(question_body.split())
        answer_text = " ".join(answer_text.split())

        inputs = self.tokenizer.encode_plus(
            question_title + " " + question_body,
            answer_text,
            add_special_tokens=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = self.max_length - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }