import lightning.pytorch as pl
from torch.utils.data import DataLoader

from get_transform_alb import get_transform
from image_dataset import ImageDataset


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(ImageDataModule, self).__init__()

        # configs
        self.cfg = cfg

        self.dataset = {}
        self.dataset['train'] = ImageDataset(
            cfg.Data.dataset.top_dir,
            cfg.Data.dataset.train_datalist,
            transform=get_transform(cfg.Transform['train'], cfg.Transform.replay))
        self.dataset['valid'] = ImageDataset(
            cfg.Data.dataset.top_dir,
            cfg.Data.dataset.valid_datalist,
            transform=get_transform(cfg.Transform['valid'], cfg.Transform.replay))
        if self.cfg.General.mode == "predict":
            self.dataset['predict'] = ImageDataset(
                cfg.Data.dataset.top_dir,
                cfg.Data.dataset.predict_datalist,
                transform=get_transform(cfg.Transform['predict'], cfg.Transform.replay))

    # call once from main process
    def prepare_data(self):
        pass
 
    # call from Trainer.fit() and Trainer.test()
    def setup(self, stage=None):
        pass
    
    # call in Trainer.fit()
    def train_dataloader(self):
        train_loader = DataLoader(
            self.dataset['train'],
            batch_size=self.cfg.Data.dataloader.train.batch_size,
            shuffle=self.cfg.Data.dataloader.train.shuffle,
            num_workers=self.cfg.Data.dataloader.train.num_workers,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True
        )
        return train_loader

    # call in Trainer.fit() and Trainer.validate()
    def val_dataloader(self):
        val_loader = DataLoader(
            self.dataset['valid'],
            batch_size=self.cfg.Data.dataloader.valid.batch_size,
            shuffle=self.cfg.Data.dataloader.valid.shuffle,
            num_workers=self.cfg.Data.dataloader.valid.num_workers,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False
        )
        return val_loader

    # call in Trainer.predict()
    def predict_dataloader(self):
        if self.cfg.General.mode == "predict":
            predict_loader = DataLoader(
                self.dataset['predict'],
                batch_size=self.cfg.Data.dataloader.predict.batch_size,
                shuffle=self.cfg.Data.dataloader.predict.shuffle,
                num_workers=self.cfg.Data.dataloader.predict.num_workers,
                pin_memory=False,
                persistent_workers=False,
                drop_last=False
            )
        else:
            predict_loader = None
        return predict_loader
