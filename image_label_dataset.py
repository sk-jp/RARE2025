from __future__ import print_function, division

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import Dataset


class ImageLabelDataset(Dataset):
    """ Dataset for image and label """
    def __init__(self, topdir, csv_file, transform=None):
        """
        Args:
            topdir (str): Path to the top dir.
            csv_file (str): Path to the csv file with annotations.
            transform (obj): Transform objects.
        """
        super(ImageLabelDataset, self).__init__()
        
        self.topdir = topdir
        self.data_list = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        image_file = f"{self.topdir}/{self.data_list.iloc[idx, 0]}"
        image = np.array(Image.open(image_file))
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=2)
        label = self.data_list.iloc[idx, 1]

        # preprocessing for images
        if self.transform:
            image = self.transform(image=image)["image"]

        # return values
        ret = {}
        ret['image'] = image
        ret['label'] = label
        ret['filename'] = image_file

        return ret


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from read_yaml import read_yaml
    from get_transform_alb import get_transform
    import cv2

    csv_file = 'datalist/train_datalist.csv'
    yaml_file = 'yaml/ae.yaml'

    conf = read_yaml(yaml_file)    

    datadir = conf.Data.dataset.top_dir
    print("datadir:", datadir)
    
    aug = get_transform(conf.Transform['train'], replay=True)

    bs = 8
    dataset = ImageLabelDataset(datadir, csv_file, transform=aug)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)

    for batch_idx, batch in enumerate(dataloader):
        image = batch['image']
        label = batch['label']
        
        print("image:", image.shape)
        print("label:", label)

        # observe a batch and stop.
        import cv2
        for b in range(bs):
            cv2.imshow("img", image[b].numpy().transpose((1,2,0))[:,:,::-1])
            cv2.waitKey(1000)

        break

