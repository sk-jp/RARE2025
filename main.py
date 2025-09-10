import argparse
from datetime import datetime
import glob
import os

from lightning.pytorch.loggers import WandbLogger
#from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.tuner import Tuner
from custom_early_stopping import CustomEarlyStopping

from typing import Dict
from pathlib import Path
import shutil
import warnings
import torch

from lightning_module import LightningModule
from image_label_datamodule import ImageLabelDataModule
from image_datamodule import ImageDataModule
from read_yaml import read_yaml

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')
torch.multiprocessing.set_sharing_strategy('file_system')


class LitProgressBar(TQDMProgressBar):
    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
#        bar.bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} [LAP:{elapsed}/ETA:{remaining}{postfix}]'
        bar.disable = True
        return bar
    
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} [LAP:{elapsed}/ETA:{remaining}{postfix}]'
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} [LAP:{elapsed}/ETA:{remaining}{postfix}]'
#        bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} [LAP:{elapsed}/ETA:{remaining}{postfix}]'
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        bar.bar_format = '{l_bar}{bar}|{n_fmt}/{total_fmt} [LAP:{elapsed}/ETA:{remaining}{postfix}]'
#        bar.disable = True
        return bar


class InitialLearningRateFinder(LearningRateFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--config", required=True, type=str, help="Path to config file")
    arg("--gpus", default="0", type=str, help="GPU IDs")
    arg("--debug", action="store_true", help="Debug mode")
    arg("--predict", action="store_true", help="Run prediction")
    arg("--save_results", action="store_true", help="save prediction results")
    arg("--lr_tune", action="store_true", help="Tune learning rate")
    arg("--lr", default=None, type=float, help="Initial learning rate")
    arg("--seed", default=None, type=int, help="Random seed")
    arg("--rm_cache_dir", action="store_true", help="Remove data cache directory")
    arg("--outdir_ext", default=None, type=str, help="Extention of name of output directory")

    return parser


def train(cfg_name: str, cfg: Dict, output_path: Path) -> None:
    """ Training main function
    """
    
    # == initial settings ==
    # random seed
    if isinstance(cfg.General.seed, int):
        seed_everything(seed=cfg.General.seed, workers=True)

    # Patch for "RuntimeError: received 0 items of ancdata" error
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # debug flag
    debug = cfg.General.debug

    # logger
    if cfg.General.mode == "validate" or cfg.General.mode == "predict":
        logger = []
    else:
        wandb_logger = WandbLogger(
            project=cfg.General.project,
            name=f"{cfg_name}",
            offline=debug)
    
        logger = [wandb_logger]

    # == callbacks ==
    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_path),
        filename=cfg.Model.arch + "-{epoch:02d}-{valid_loss:.2f}",
        monitor='valid_loss',
        save_weights_only=True,
        save_top_k=1,
        mode='min'
    )

    # learning rate monitor
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    # progress bar
    progressbar_callback = LitProgressBar()
    # early stopping
    early_stop_callback = CustomEarlyStopping(
        monitor=cfg.Earlystop.monitor,
        mode=cfg.Earlystop.mode,
        patience=cfg.Earlystop.patience,
        verbose=cfg.Earlystop.verbose        
    )

    # == plugins ==
    plugins = None
#    if not cfg.General.lr_tune:
    if cfg.General.strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False)
#        if cfg.Model.arch == 'dinov2':
#            strategy = DDPStrategy(find_unused_parameters=True)
    elif cfg.General.strategy == "fsdp":
        strategy = FSDPStrategy()

#    callbacks = [checkpoint_callback, lr_monitor_callback, progressbar_callback, early_stop_callback, image_save_callback]
    callbacks = [checkpoint_callback, lr_monitor_callback, progressbar_callback, early_stop_callback]

    # == Trainer ==
    default_root_dir = os.getcwd()
    trainer = Trainer(
        accelerator=cfg.General.accelerator,
        strategy=strategy,
        devices=cfg.General.gpus if cfg.General.accelerator == "gpu" else None,
        num_nodes=cfg.General.num_nodes,
        precision=cfg.General.precision,
        logger=logger,
        callbacks=callbacks,
        max_epochs=6 if debug else cfg.General.epoch,
        limit_train_batches=0.1 if debug else 1.0,
        limit_val_batches=1.0 if debug else 1.0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg.General.check_val_every_n_epoch,
        accumulate_grad_batches=cfg.Optimizer.accumulate_grad_batches,
        deterministic=False,
        benchmark=False,
        plugins=plugins,
        default_root_dir=default_root_dir,
    )

    # Lightning module and data module
    model = LightningModule(cfg)

    if cfg.General.mode == "train":
        datamodule = ImageLabelDataModule(cfg)
    elif cfg.General.mode == "predict" or cfg.General.mode == "test": 
        datamodule = ImageDataModule(cfg)
    
    if cfg.General.lr_tune:
        # Run learning rate finder
        print('*** Start learning rate finder ***')
        tuner = Tuner(trainer)

        lr_finder = tuner.lr_find(model, datamodule)
                                   #, attr_name="model.cfg.Optimizer.optimizer.params.lr")
#            min_lr=1e-6, max_lr=1, num_training=100, mode='linear', step_mode='exp')
        
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # suggested LR
        new_lr = lr_finder.suggestion()
        print('Suggested LR:', new_lr)
        
        # remove temporal checkpoint files
        for p in glob.glob(f'{default_root_dir}/lr_find_temp_model_*.ckpt'):
            if os.path.isfile(p):
                os.remove(p)               
    elif cfg.General.mode == "validate":
        # run prediction
        print('*** Start validation ***')
        trainer.validate(model, datamodule=datamodule)
    elif cfg.General.mode == "predict":
        # run prediction
        print('*** Start prediction ***')
        trainer.predict(model, datamodule=datamodule)
    elif cfg.General.mode == "test":
        # run test
        print('*** Start test ***')
        trainer.test(model, datamodule=datamodule)
    else:
        # Start training
        print('*** Start training ***')
        trainer.fit(model, datamodule=datamodule)

def make_directory(path, remove_dir=False):
    """ Make a directory
        Args:
            path (str): path of the directory to make
            remove_dir (bool): remove the directory if it exists when True
    """    
    if os.path.exists(path):
        if remove_dir is True:
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

def main():
    # parse args
    parser = make_parse()
    args = parser.parse_args()
    print('args:', args)

    # Read config
    cfg = read_yaml(fpath=args.config)
    if args.gpus is not None:
        cfg.General.gpus = list(map(int, args.gpus.split(",")))
    if cfg.General.debug or args.debug:
        cfg.General.debug = True
    else:
        cfg.General.debug = False
    if cfg.General.lr_tune or args.lr_tune:
        cfg.General.lr_tune = True
    else:
        cfg.General.lr_tune = False
    if args.predict:
        cfg.General.mode = "predict"        
    if args.save_results or cfg.General.save_results:
        assert(cfg.General.mode == "predict"), f"mode is {cfg.General.mode}"
        cfg.General.save_results = True        
    if args.lr is not None:
        cfg.Optimizer.optimizer.params.lr = args.lr
    if args.seed is not None:
        cfg.General.seed = args.seed

    # check args and configs  
    assert (cfg.General.mode == "train" or cfg.General.mode == "validate" \
        or cfg.General.mode == "predict" or cfg.General.mode == "test"), \
            f"cfg.General.mode = {cfg.General.mode}"
    
    if args.rm_cache_dir:
        if os.path.exists(cfg.Data.dataset.cache_dir):
            # remove chche dir
            print(f"Remove the data cache dir: {cfg.Data.dataset.cache_dir}")
            shutil.rmtree(cfg.Data.dataset.cache_dir)
   
    # Make output path and dir
    path_str = datetime.now().strftime("%y%m%d_%H%M%S_")
    ext_str = ''
    config_name = os.path.basename(args.config).split(".")[0]
    ext_str += f'{config_name}'
    if args.lr is not None:
        ext_str += f'-LR{args.lr}'
    if args.seed is not None:
        ext_str += f'-SEED{args.seed}'
    if cfg.General.mode == "test":
        ext_str += f'-test'
    if cfg.General.mode == "predict":
        ext_str += f'_predict'
    if args.outdir_ext is not None:
        ext_str += f'-{args.outdir_ext}'
    path_str += ext_str
    cfg_name = ext_str
    
    output_path = Path('./results') / Path(config_name) / Path(path_str)
    print('output_path: ', output_path)
    make_directory(output_path, remove_dir=False)
    cfg.General.output_path = output_path
    # Config and Source code backup
    shutil.copy2(args.config, str(output_path / Path(args.config).name))

    # Start train/valid or predict
    train(cfg_name=cfg_name, cfg=cfg, output_path=output_path)


if __name__ == '__main__':
    main()
