from torch import nn
import torch
from tqdm import tqdm



class down(nn.Module):
    
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.couche1 = nn.Conv2d(in_channels=in_channels, out_channels= out_channels,kernel_size=3, stride=2, padding=1)
        self.couche2 = nn.Conv2d(in_channels=out_channels, out_channels= out_channels,kernel_size=3, stride=2, padding=1)
        
    def forward(self,x):
        x = self.couche1(x)
        x = self.nn.functional.leaky_relu(x)
        x = self.couche2(x)
        return nn.functional.leaky_relu(x)
    


class up(nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super().__init__()
        self.activation = activation # Pour la dernière couche, on veut des valeurs entre -1 et 1
        self.couche = nn.ConvTranspose2d(in_channels=in_channel,out_channels=out_channel, kernel_size=(2,2), stride=2, padding='same')
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.conv2d = nn.Conv2d(in_channels=out_channels, out_channels= out_channels,kernel_size=(2,2),  padding='same')
        
    def forward(self,x, connection, activation = True):
        
      x = self.couche(x)
      x = self.conv2d(torch.cat([x , connection],dim=1))
      x = nn.functional.leaky_relu(x)
      x = self.conv2d(torch.cat([x , connection],dim=1))
      if activation :
        x = nn.functional.leaky_relu(x)
      else:
        x = 2* torch.nn.functional.sigmoid(x) -1
      return x
    

class Unet_with_add(nn.Module):
    def __init__(self,nb_features) -> None:
        super().__init__()
        self.down1 = down(1,nb_features)
        self.down2 = down(nb_features,nb_features*2)
        self.down3 = down(nb_features * 2, nb_features* 4)

        self.down4 = down(nb_features* 4, nb_features *8)

        self.up1 = up(nb_features *8, nb_features*4, apply_dropout=True)
        self.up2 = up(nb_features *4, nb_features*2)
        self.up3 = up(nb_features *2, nb_features)
        self.up4 = up(nb_features, 3, False)
    
    def forward(self,x):
        x = self.down1(x) # (1,64,64)
        x1 = x
        x = self.down2(x)
        x2 = x
        x = self.down3(x)
        x3 = x
        x = self.down4(x)
        x = self.up1(x)
        x = self.up2(x + x3)
        x = self.up3(x + x2)
        x = self.up4(x + x1) # (3,64,64)
        return x
    
    def loss(self,real_images, fake_images, disc_pred,l1_loss, bce_loss):
        l1=l1_loss(real_images,fake_images)
        dupage_discriminateur = bce_loss(torch.ones_like(disc_pred), disc_pred)
        loss_gen = dupage_discriminateur + 100 * l1
        return loss_gen
    
class Unet_with_cat(nn.Module):
    def __init__(self,nb_features) -> None:
        super().__init__()
        self.down1 = down(1,nb_features) 
        self.down2 = down(nb_features,nb_features*2) 
        self.down3 = down(nb_features * 2, nb_features* 4) 

        self.down4 = down(nb_features* 4, nb_features *8) 

        self.up1 = up(nb_features *8, nb_features*4) 
        self.up2 = up(nb_features *4*2, nb_features*2) 
        self.up3 = up(nb_features *2*2, nb_features) 
        self.up4 = up(nb_features*2, 3) 
    
    def forward(self,x):
        x = self.down1(x) 
        x1 = x
        x = self.down2(x) 
        x2 = x
        x = self.down3(x) 
        x3 = x
        x = self.down4(x) 
        x = self.up1(x) 
        x = self.up2(x,x3) 
        x = self.up3(x,x2) 
        x = self.up4(x,x1, activation = False) 
        return x
    
    def loss(self,real_images, fake_images, disc_pred,l1_loss, bce_loss):
        l1=l1_loss(real_images,fake_images)
        dupage_discriminateur = bce_loss(torch.ones_like(disc_pred), disc_pred)
        loss_gen = dupage_discriminateur + 100 * l1
        return loss_gen
class Discriminateur(nn.Module):
    
    #Ne marche probablement pas, je n'ai pas réfléchi aux tailles et nombre de canaux
    def __init__(self, nb_features) -> None:
        super().__init__()
        self.down1 = down(4,nb_features)
        self.down2 = down(nb_features,nb_features*2)
        self.down3 = down(nb_features * 2, nb_features* 4)
        self.down4 = down(nb_features * 4, nb_features* 8)
        self.down5 = down(nb_features * 8, 1) # On veut une grille de pixels
        # En sortie on n'a pas un nombre car Patch Gan !!! Chaque pixel de l'image 
    def forward(moi, x, y):
        x = torch.cat([x,y], dim = 1) # (batch size, 256,256,channels*2)
        x = moi.down1(x) 
        x = moi.down2(x)
        x = moi.down3(x)
        x = moi.down4(x)
        x = moi.down5(x)
        return x
    def loss(self,disc_real_output, disc_generated_output, bce_loss):
        
        real_loss = bce_loss(torch.ones_like(disc_real_output),disc_real_output)
        generated_loss = bce_loss(torch.zeros_like(disc_generated_output),disc_generated_output)
        
        total_disc_loss= real_loss + generated_loss
        
        return total_disc_loss
    
class Pix2Pix(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.unet = Unet_with_cat(64)
        self.discriminateur = Discriminateur(32)
        self.optimizer_unet = torch.optim.Adam(self.unet.parameters())
        self.optimizer_discriminateur = torch.optim.Adam(self.discriminateur.parameters())
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.l1_loss = torch.nn.L1Loss()
        
    def train_step(self, X_batch, Y_batch):
        
        # On commence par entrainer le discriminateur
        self.discriminateur.zero_grad()
        
        fake_images = self.unet(X_batch)
        real_images = Y_batch
        disc_real_output = self.discriminateur(real_images,X_batch)
        disc_fake_output = self.discriminateur(fake_images,X_batch)
        gan_loss = self.discriminateur.loss(disc_real_output, disc_fake_output, self.bce_loss)
        
        gan_loss.backward()
        self.optimizer_discriminateur.step()
        
        # On entraine maintenant le générateur
        self.unet.zero_grad()
        fake_images = self.unet(X_batch)
        disc_fake_output = self.discriminateur(fake_images,X_batch)

        gen_loss = self.unet.loss(Y_batch, fake_images, disc_fake_output, self.l1_loss, self.bce_loss)
        
        loss_value = gen_loss.item()
        
        gen_loss.backward()
        self.optimizer_unet.step()
        return loss_value
    
    def train(self, epoch, dataloader):
        
        for e in tqdm(range(epoch)):
            
            progression = tqdm(dataloader, colour="#f0768b")
            for i, batch in enumerate(progression):
                loss = self.train_step(batch[0], batch[1])
                progression.set_description(f"Epoch {e+1}/{epoch} | loss_gen: {loss}")

    def forward(self,x):
        return self.unet.forward(x)
    