from torch.utils.data import Dataset
from torchvision.transforms import v2
import pandas as pd
import torchvision
from torchvision.io import read_image, ImageReadMode
#On crée un dataset custom à partir des images du dossier 'dataset'


class TrainDataset(Dataset):
    def __init__(self, img_csv, img_dir, img_size, dataset_size, train_dataset_size=-1):
        self.img_dir = img_dir
        self.img_csv = pd.read_csv(img_csv, skipfooter = int(dataset_size*0.1),engine = "python") # L'argument skipfooter permet de sauter les x dernières lignes du fichier et de les réserver pour le jeu de test
        self.transform = v2.RandomCrop((256,256))
        self.train_dataset_size = train_dataset_size
        self.img_size= img_size
    def __len__(self):
        if self.train_dataset_size == -1:
            return len(self.img_csv)
        else:
            return self.train_dataset_size

    def __getitem__(self, index):
        ### A compléter ###
        img_path_wb= self.img_csv.iloc[index,1]
        
        img_path_couleur =self.img_csv.iloc[index,0]
        wb, couleur =read_image(img_path_wb, mode = ImageReadMode.GRAY ), read_image(img_path_couleur)
        i,j,h,w = self.transform.get_params(wb, output_size= self.img_size)

        wb, couleur= torchvision.transforms.functional.crop(wb,i,j,h,w), torchvision.transforms.functional.crop(couleur,i,j,h,w)

        wb, couleur= (wb -127.5)/127.5, (couleur -127.5)/127.5
        return wb, couleur

        # On veut retourner un élément du dataset à partir de son indice
        # On veut également normaliser les éléments de notre dataset entre -1 et 1
     
class TestDataset(Dataset):
    def __init__(self, img_csv, img_dir, img_size, dataset_size, train_dataset_size=-1):
        self.img_dir = img_dir
        self.img_csv = pd.read_csv(img_csv, skiprows= int(dataset_size*0.9),engine = "python") #Taille du dataset - taille du test )
        self.transform = torchvision.transforms.Resize(img_size)
        #self.transform = v2.RandomCrop((256,256))
        self.train_dataset_size = train_dataset_size
        self.img_size= img_size
    def __len__(self):
        if self.train_dataset_size == -1:
            return len(self.img_csv)
        else:
            return self.train_dataset_size

    def __getitem__(self, index):
        ### A compléter ###
        img_path_wb= self.img_csv.iloc[index,1]
        
        img_path_couleur =self.img_csv.iloc[index,0]
        wb, couleur =read_image(img_path_wb, mode = ImageReadMode.GRAY ), read_image(img_path_couleur)

        wb, couleur= (wb -127.5)/127.5, (couleur -127.5)/127.5
        return wb, couleur

        # On veut retourner un élément du dataset à partir de son indice
        # On veut également normaliser les éléments de notre dataset entre -1 et 1   
