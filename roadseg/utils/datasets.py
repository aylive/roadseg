
import cv2
import pandas as pd
from torch.utils.data import Dataset

from utils.utils import *

class SatImage(Dataset):
    """
    Read images, apply augmentation and preprocessing transformations.
    """
    def __init__(
        self,
        meta_path,
        class_rgb_values=None,
        augmentation=None,
        preprocessing=None,
    ):
        metadata = pd.read_csv(meta_path)
        self.image_path = metadata['sat_img_pth'].tolist()
        self.mask_path = metadata['mask_pth'].tolist()
        
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # read image
        image = cv2.cvtColor(cv2.imread(self.image_path[i]), cv2.COLOR_BGR2RGB) # bgr2rgb?
        mask = cv2.cvtColor(cv2.imread(self.mask_path[i]), cv2.COLOR_BGR2RGB)
        
        # image preprocess - resize
        if image.shape[:2] != (352, 352):
            image = cv2.resize(image, dsize=(352, 352), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, dsize=(352, 352), interpolation=cv2.INTER_AREA)
        
        # image preprocess - one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float32')
        
        # image preprocess - augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask) # ???
            image, mask = sample['image'], sample['mask']
        
        # other preprocesses
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
    
    def __len__(self):
        return len(self.image_path)