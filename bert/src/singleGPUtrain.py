import torch.nn as nn
import torch
import transformers
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AdamW,get_linear_schedule_with_warmup
from scipy import stats 
import warnings
from dataset import BertDatasetTrainning
from models import BERTBasedUncased
warnings.filterwarnings('ignore')

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
        optimizer.step() #optimizer.step() # when only one node  ,barrier = True
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
        

def main():
    MAX_LEN = 512
    BATCH_SIZE = 8
    EPOCH = 100

    df = pd.read_csv('data/train.csv').fillna('none')
    dftrain ,dftest = train_test_split(df, random_state = 42, test_size = 0.1)

    dftrain = dftrain.reset_index(drop=True)
    dftest = dftest.reset_index(drop=True)

    samplesubmission = pd.read_csv('data/sample_submission.csv')
    target_cols = list(samplesubmission.drop('qa_id',axis = 1).columns)

    train_target = dftrain[target_cols].values
    test_target = dftest[target_cols].values
    

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased/')
    train_dataset = BertDatasetTrainning(
        title = dftrain.question_title.values,
        body = dftrain.question_body.values,
        answer = dftrain.answer.values,
        targets = train_target,
        tokenizer = tokenizer,
        max_len = MAX_LEN
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    valid_dataset = BertDatasetTrainning(
        title = dftest.question_title.values,
        body = dftest.question_body.values,
        answer = dftest.answer.values,
        targets = test_target,
        tokenizer = tokenizer,
        max_len = MAX_LEN
    )


    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True 
    )

    ### Model

    device ='cuda' if torch.cuda.is_available() else 'cpu'
    lr = 2e-5
    model = BERTBasedUncased('bert-base-uncased/').to(device)
    num_train_steps = int(len(train_dataset)/BATCH_SIZE * EPOCH)

    optimizer = AdamW(model.parameters(),lr =lr)


    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_train_steps
    )

    for epoch in range(EPOCH):
        training_loop_fn(train_loader,model,optimizer,device,scheduler)

        o,t = eval_loop_fn(valid_loader,model,device)

        spear = []
        for jj in range(t.shape[1]):
            p1 = list(t[:,jj])
            p2 = list(o[:,jj])
            coef, _ = np.nan_to_num(stats.spearmanr(p1,p2))
            spear.append(coef)

        spear = np.mean(spear)
        print('In the {} Spearman is{}'.format(epoch,spear))
        torch.save(model.state_dict(),'model.bin')


if __name__ =='__main__':
   main()
