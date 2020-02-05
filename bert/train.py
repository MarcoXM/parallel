import torch.nn as nn
import torch
import transformers
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AdamW,get_linear_schedule_with_warmup
import torch_xla.core.xla_model as xm
from scipy import stats 
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import warnings
warnings.filterwarnings('ignore')


class BERTBasedUncased(nn.Module):
    def __init__(self,bert_path):
        super(BERTBasedUncased,self).__init__()

        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path) # Load the model from pretrained
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768,30) # Getting 30 dimension ouput as result

    def forward(self,ids, mask, token_type_ids):
        _, output2 = self.bert(ids, attention_mask = mask,token_type_ids=token_type_ids) # Bert have 2 output and the second one is our need
        finaloupt = self.out(output2)
        return finaloupt




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
            add_special_token = True,
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


def loss_fn(output,target):
    return nn.BCEWithLogitsLoss()(output,target)


def training_loop_fn(data_loader, model, optimizer, device, scheduler = None):
    model.train()
    losses = []

    for bi,data in enumerate(data_loader):
        ids = data['ids'].to(device,dtype = torch.long)
        mask = data['mask'].to(device,dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device,dtype = torch.long)
        target = data['targets'].to(device,dtype = torch.float)

        optimizer.zero_grad()
        output = model(ids=ids,mask=mask,token_type_ids=token_type_ids)
        loss = loss_fn(output,target)
        losses.append(loss.item())
        loss.backward()
        xm.optimizer_step(optimizer) #optimizer.step() # when only one node  ,barrier = True
        if scheduler is not None:
            scheduler.step()


def eval_loop_fn(data_loader, model,device,):
    model.train()
    fin_target = []
    fin_output = []

    for bi,data in enumerate(data_loader):
        ids = data['ids'].to(device,dtype = torch.long)
        mask = data['mask'].to(device,dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device,dtype = torch.long)
        target = data['targets'].to(device,dtype = torch.float)


        output = model(ids=ids,mask=mask,token_type_ids=token_type_ids)
        loss = loss_fn(output,target)
        
        fin_output.append(output.cpu().detach().numpy())
        fin_target.append(target.cpu().detach().numpy())

    return np.vstack(fin_output),np.vstack(fin_target)
        

def main(index):
    MAX_LEN = 512
    BATCH_SIZE = 4
    EPOCH = 20

    df = pd.read_csv('data/train.csv').fillna('none')
    dftrain ,dftest = train_test_split(df, random_state = 42, test_size = 0.1)

    dftrain = dftrain.reset_index(drop=True)
    dftest = dftest.reset_index(drop=True)

    samplesubmission = pd.read_csv('data/sample_submission.csv')
    target_cols = list(samplesubmission.drop('qa_id',axis = 1).columns)

    train_target = dftrain[target_cols].values
    test_target = dftest[target_cols].values
    

    tokenizer = transformers.BertTokenizer.from_pretrained('bert_based_uncased')
    train_dataset = BertDatasetTrainning(
        title = dftrain.question_title.values,
        body = dftrain.question_body.values,
        answer = dftrain.answer.values,
        targets = train_target,
        tokenizer = tokenizer,
        max_len = MAX_LEN
    )

    trainsampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank = xm.get_ordinal(),
        shuffle = True  
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        sampler = trainsampler
    )

    valid_dataset = BertDatasetTrainning(
        title = dftest.question_title.values,
        body = dftest.question_body.values,
        answer = dftest.answer.values,
        targets = test_target,
        tokenizer = tokenizer,
        max_len = MAX_LEN
    )
    validsampler = torch.utils.data.DistributedSampler(
        valid_dataset,
        num_replicas=xm.xrt_world_size(),
        rank = xm.get_ordinal()
    )


    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = BATCH_SIZE,
        sampler = validsampler  
    )

    ### Model

    device = xm.xla_device()
    lr = 3e-5 * xm.xrt_world_size()
    model = BERTBasedUncased('bert_based_uncased').to(device)
    num_train_steps = int(len(train_dataset)/BATCH_SIZE/xm.xrt_world_size() * EPOCH)

    optimizer = AdamW(model.parameters(),lr =lr)


    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_train_steps
    )

    for epoch in range(EPOCH):

        paraloader = pl.ParallelLoader(train_loader,[device])
        training_loop_fn(paraloader.per_decice_loader(device),model,optimizer,device,scheduler)

        paraloader = pl.ParallelLoader(valid_loader,[device])
        o,t = eval_loop_fn(paraloader.per_decice_loader(device),model,device)

        spear = []
        for jj in range(t.shape[1]):
            p1 = list(t[:,jj])
            p2 = list(o[:,jj])
            coef, _ = np.nan_to_num(stats.spearmanr(p1,p2))
            spear.append(coef)

        spear = np.mean(spear)
        xm.master_print('Spearman is',spear)
        xm.save(model.state_dict(),'model.bin')


if __name__ =='__main__':
    xmp.spawn(main,nprocs = 8)