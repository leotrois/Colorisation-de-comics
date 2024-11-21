from torch import cat, autograd, unsqueeze, squeeze, permute
from torch.utils.data import DataLoader
from classes import CustomDataset
from models import Pix2Pix
import torch
import pandas as pd
import matplotlib.pyplot  as plt
from torchvision.utils import make_grid, save_image


BATCH_SIZE = 32
IMG_SIZE = (512,512)

data = CustomDataset("dataset.csv", IMG_SIZE)

train_size = int(0.9 * len(data))
test_size = len(data) - train_size



train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

x_train = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
x_test = DataLoader(train_dataset, 1, shuffle = True)

device = "cuda"
model = Pix2Pix().to(device)
for batch in x_test:
    batch_test = batch
    break


grid =permute( make_grid([batch[1][0],squeeze(model.forward(unsqueeze(batch[0][0],dim = 0)))]), (1,2,0))
plt.imshow(grid.to('cpu'))
plt.savefig(f'test_value_{-1}_epoch.png')

for i in range(10):
    model.train(1, x_train)



    grid =permute( make_grid([batch[1][0],squeeze(model.forward(unsqueeze(batch[0][0],dim = 0)))]), (1,2,0))
    plt.imshow(grid.to('cpu'))
    plt.savefig(f'test_value_{i}_epoch.png')

    torch.save(model.state_dict(), f"weights_epoch{i}.pt")