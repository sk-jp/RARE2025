from __future__ import print_function, division

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """ Dataset for image and label """
    def __init__(self, topdir, csv_file, transform=None):
        """
        Args:
            topdir (str): Path to the top dir.
            csv_file (str): Path to the csv file with annotations.
            transform (obj): Transform objects.
        """
        super(ImageDataset, self).__init__()
        
        self.topdir = topdir
        self.data_list = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        case_id = self.data_list.iloc[idx, 0]
        image_dir = f"{self.topdir}/frames"
        images = []
        frame_numbers = []
        for num in range(0, 2700, 30):
            image_file = f"{image_dir}/{case_id}_{num}.jpg"
            image = np.array(Image.open(image_file))

            # preprocessing for images
            if self.transform:
                if num == 0:
                    data = self.transform(image=image)
                    images.append(data['image'].unsqueeze(0))
                else:
                    data2 = A.ReplayCompose.replay(data['replay'], image=image)
                    images.append(data2['image'].unsqueeze(0))
            else:
                raise NotImplementedError

            frame_numbers.append(num)

        images = torch.cat(images, dim=0) 
        frame_numbers = torch.tensor(frame_numbers)

        # return values
        ret = {}
        ret['image'] = images
        ret['case_id'] = case_id
        ret['frame_number'] = frame_numbers

        return ret


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from read_yaml import read_yaml
    from get_transform_alb import get_transform
    import cv2

    csv_file = 'datalist/train_datalist.csv'
    yaml_file = 'yaml/resnet50.yaml'

    conf = read_yaml(yaml_file)    

    datadir = conf.Data.dataset.top_dir
    print("datadir:", datadir)
    
#    print("trans:", conf.Transform['train'])
    aug = get_transform(conf.Transform['train'], replay=True)

    bs = 8
    dataset = ImageLabelDataset(datadir, csv_file, transform=aug)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)

    for batch_idx, batch in enumerate(dataloader):
        image = batch['image']
        label = batch['label']
        
        print("image:", image.shape)
        print("label:", label.shape)

        # observe a batch and stop.
        import cv2
        for b in range(bs):
            cv2.imshow("img", image[b][0].numpy().transpose((1,2,0))[:,:,::-1])
            cv2.waitKey(1000)

        break

