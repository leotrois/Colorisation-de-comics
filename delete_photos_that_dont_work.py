import os
from tqdm import tqdm
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
PATH = "/raid/home/automatants/soudre_leo/Mini_projet_Automatants/Dataset"

i= 0
def delete_photos_that_dont_load(path):
    image = cv.imread(path)
    if image is None:
        global i
        os.remove(path)
        print(f"Deleted {path}")
        i = i +1

def process_volume(vol):
    vol_path = os.path.join(PATH, vol)
    for folders in os.listdir(vol_path):
        for image_name in os.listdir(os.path.join(vol_path,folders)):
            delete_photos_that_dont_load(os.path.join(vol_path,folders, image_name))


volumes = os.listdir(PATH)
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_volume, volumes), total=len(volumes)))
    
