from torch import cat, autograd, unsqueeze, squeeze, permute
from torch.utils.data import DataLoader
from classes import TrainDataset, TestDataset
from models import Pix2Pix
import pandas as pd
import matplotlib.pyplot  as plt
from torchvision.utils import make_grid, save_image


train = TrainDataset("dataset.csv", "./Dataset", (256,256), len(csv))
test = TestDataset("dataset.csv", "./Dataset", (256,256), len(csv))


data =  DataLoader(train, batch_size= 512, shuffle= True)
data_test = DataLoader(test, batch_size= 1, shuffle= True)
for batch in data:
    print("batch",len(batch))
    break
device = "cuda"
model = Pix2Pix().to(device)

for i in range(10):
    model.train(1, data)

    for batch in data_test:
        batch_test = batch
        break

    grid =permute( make_grid([batch[1][0],squeeze(model.forward(unsqueeze(batch[0][0],dim = 0)))]), (1,2,0))
    print(grid.size())
    plt.imshow(grid)
    plt.savefig(f'test_value_{i}_epoch.png')
