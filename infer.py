from models import Pix2Pix
from torchvision.io import read_image
from torchvision import transforms
from torch import permute
from torchvision.utils import save_image


PRETRAINED = True
IMG_SIZE = (512, 512)


if PRETRAINED:
    PATH = "Pix2Pix_weights_10_epochs.pt"
else:
    PATH = "Weights_trained.pt"
    
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
    
if __name__ == "__main__":
    model = Pix2Pix()
    model.load_state_dict(PATH, weights_only=True)
    model.eval()
    
    model.to(device)
    
    image = read_image('./infer_image.jpg')
    image = transforms.Resize(IMG_SIZE)(image)
    image = (image - 127.5) / 127.5
    image = image.unsqueeze(0).to(device)
    image = model.forward(image)
    image = squeeze(image)
    image = permute(image, (1,2,0))

    save_image(image,"./result.jpg")
    