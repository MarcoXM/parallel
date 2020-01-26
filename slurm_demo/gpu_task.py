import torch
import torch.nn as nn
from datasetdemo import RandomDataset
from models import SingleModel,MultiModel








if __name__ == "__main__":


    BATCHSIZE = 64
    INPUT_SIZE = 30000
    OUTPUT_SIZE = 2
    DATA_SIZE = 784

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data and Model
    dataset = RandomDataset(DATA_SIZE,INPUT_SIZE)
    model = SingleModel(DATA_SIZE,512,256,OUTPUT_SIZE)
    
    rand_loader = torch.utils.data.DataLoader(dataset =dataset,
    batch_size=BATCHSIZE)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

    model.to(device)
    model.train()
    for data in rand_loader:
        inputs = data.to(device)
        outputs = model(inputs)
        print("Outside: input size", inputs.size(),
            "output_size", outputs.size())