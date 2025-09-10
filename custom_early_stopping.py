from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class CustomEarlyStopping(EarlyStopping):
    def on_train_end(self, trainer, pl_module):
        if self.stopped_epoch != 0:
            print(f"\n[EarlyStopping] Triggered at epoch {self.stopped_epoch}:"
                  f" {self.monitor} did not improve for {self.patience} epochs.\n")
        super().on_train_end(trainer, pl_module)
