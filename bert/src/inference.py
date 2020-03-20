import torch.nn as nn
import torch
import transformers
import pandas as pd 
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split
from transformers import AdamW,get_linear_schedule_with_warmup
from scipy import stats 
import warnings
from scipy import stats
from dataset import BertDatasetTrainning,BERTDatasetTest
from models import BERTBasedUncased
import glob
warnings.filterwarnings('ignore')


def predict(test_batch_size,device, test_data,models_path ):
    DEVICE = device
    TEST_BATCH_SIZE = test_batch_size
    TEST_DATASET = test_data
    df = pd.read_csv(TEST_DATASET).fillna("none")

    qtitle = df.question_title.values.astype(str).tolist()
    qbody = df.question_body.values.astype(str).tolist()
    answer = df.answer.values.astype(str).tolist()
    category = df.category.values.astype(str).tolist()

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased/", 
                                                           do_lower_case=True)
    maxlen = 512
    predictions = []

    test_dataset = BERTDatasetTest(
        qtitle=qtitle,
        qbody=qbody,
        answer=answer,
        tokenizer=tokenizer,
        max_length=maxlen
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    model = BERTBasedUncased("bert-base-uncased/")
    model.to(DEVICE)
    model.load_state_dict(torch.load(models_path))
    model.eval()

    tk0 = tqdm(test_data_loader, total=int(len(test_dataset) / test_data_loader.batch_size))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]

        ids = ids.to(DEVICE, dtype=torch.long)
        mask = mask.to(DEVICE, dtype=torch.long)
        token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs = torch.sigmoid(outputs).cpu().numpy()
            predictions.append(outputs)

    return np.vstack(predictions)

if __name__ == '__main__':
    SAMPLE_SUBMISSION = "data/sample_submission.csv"
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    target_cols = list(sample.drop("qa_id", axis=1).columns)
    ans = np.zeros_like(sample[target_cols].values)

    models_p = sorted(glob.glob('BERT_FOLD_5/*.bin'))
    print(models_p)
    for i in range(len(models_p)):
        prediction = predict(test_batch_size = 4 ,device='cuda', test_data = 'data/test.csv',models_path = models_p[i] )
        ans +=prediction
        print('Get Fold {} result '.format(i))
    ans = ans/len(models_p)
    sample[target_cols] = ans
    sample.to_csv("submission_final.csv", index=False)
