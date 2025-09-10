import glob
import numpy as np
import torch
import lightning.pytorch as pl
import os
import shutil

from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelAveragePrecision

from sklearn.metrics import roc_auc_score, average_precision_score
from ppv_at_recall import ppv_at_recall

from fix_model_state_dict import fix_model_state_dict
from get_optimizer import get_optimizer
from loss import Loss

class LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super(LightningModule, self).__init__()
        self.cfg = cfg
        
        # model
        if cfg.Model.arch == 'ae':
            from ae import AE
            self.model = AE(**cfg.Model.params)
        elif cfg.Model.arch == 'convnext':
            from convnext import ConvNeXt
            self.model = ConvNeXt(**cfg.Model.params)
        else:
            raise ValueError(f'{cfg.Model.arch} is not supported.')
        
        if cfg.Model.pretrained is not None:
            # Load pretrained model weights
            print(f'Loading: {cfg.Model.pretrained}')
            checkpoint = torch.load(cfg.Model.pretrained, map_location='cpu')
            state_dict = checkpoint['state_dict']
            state_dict = fix_model_state_dict(state_dict)
            self.model.load_state_dict(state_dict)

        # output buffers
        self.training_step_outputs = []        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []

        # metrics
        self.valid_metrics_fun = [roc_auc_score, average_precision_score, ppv_at_recall]

        # flag to check the validation is performed or not at the end of epoch
        self.did_validation = False

    def setup(self, stage=None):
        if stage == "fit":
            self.lossfun = Loss(self.cfg.Loss)
            self.lossfun_valid = Loss(self.cfg.Loss)
        elif stage == "validate":
            self.lossfun_valid = Loss(self.cfg.Loss)
        elif stage == "test":
            self.lossfun_test = Loss(self.cfg.Loss)

    def forward(self, x, **kwargs):
        y = self.model(x, **kwargs)
        return y

    def on_train_epoch_start(self):
        # clear GPU cache before the validation
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()
        self.did_validation = False

    def training_step(self, batch, batch_idx):
        image = batch["image"]     # [b,c,h,w]
        label = batch["label"]     # [b,l]

        # forward
        #   AE -> returns: x_hat, z, [en1, en2, en3], [de1, de2, de3]
        out = self.forward(image)

        # loss
        if self.cfg.Model.arch == 'ae':
            loss, _ = self.lossfun(image, out)
        elif self.cfg.Model.arch == 'convnext':
            loss, _ = self.lossfun(out, label)
                
        output = {"loss": loss.item()}
        self.training_step_outputs.append(output)

        return loss

    def on_train_epoch_end(self):
        # print the results
        outputs_gather = self.all_gather(self.training_step_outputs)

        if self.trainer.is_global_zero:
            epoch = int(self.current_epoch)

            # loss
            train_loss = torch.stack([o['loss']
                                      for o in outputs_gather]).mean().detach()

            # log
            d = dict()
            d['epoch'] = epoch
            d['train_loss'] = train_loss

            print('\n Mean:')
            s = f'  Train:\n'
            s += f'    loss: {train_loss.item():.3f}'
            print(s)

            if self.did_validation:
                s = '  Valid:\n'
                s += f'    loss: {self.valid_loss:.3f}'
                s += '\n'
                s += '  '
                for key in self.valid_metrics.keys():
                    s += f'  {key}: {self.valid_metrics[key]:.3f}'
                print(s)

            self.log_dict(d, prog_bar=False, rank_zero_only=True)

        self.training_step_outputs.clear()

    def on_validation_epoch_start(self):
        # clear GPU cache before the validation
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()        
     
    def validation_step(self, batch, batch_idx):
        image = batch["image"]     # [b,c,h,w]
        label = batch["label"]     # [b,l]

        """
        # forward
        out = self.forward(image)
        """

        # forward
        if self.cfg.Transform.valid_tta:
            outputs = []
            for k in range(4):  # 0°, 90°, 180°, 270°
                # rotate the image
                rotated = torch.rot90(image, k=k, dims=(2, 3))
                out = self.forward(rotated)
                outputs.append(out)
            out = torch.stack(outputs, dim=0).mean(dim=0)
        else:
            out = self.forward(image)

        if self.cfg.Model.arch == 'ae':
            # loss
            loss, _ = self.lossfun_valid(image, out)
            # score
            score, _ = self.lossfun_valid(image, out, anomaly_score=True, keepdim=True)
            score = torch.mean(score, dim=[1, 2, 3])    # [B,1,H,W] -> [B]

            output = {"loss": loss,
                      "label": label,
                      "score": score}

        elif self.cfg.Model.arch == 'convnext':
            loss, _ = self.lossfun(out, label)

            output = {"loss": loss,
                      "score": torch.softmax(out, dim=-1)[:, 1],
                      "label": label}

#        print("output:", output['loss'].shape, output['label'].shape, output['score'].shape)

        self.validation_step_outputs.append(output)

        return
   
    def on_validation_epoch_end(self):
        # all gather
        outputs_gather = self.all_gather(self.validation_step_outputs)

        epoch = int(self.current_epoch)
            
        # loss
        valid_loss = torch.stack([o['loss'] for o in outputs_gather]).mean().item()
        self.valid_loss = valid_loss

        # metrics
        self.valid_metrics = dict()
        labels = torch.hstack([o['label'] for o in outputs_gather]).cpu().numpy()
        scores = torch.hstack([o['score'] for o in outputs_gather]).cpu().numpy()
        labels = labels.squeeze()  # [1,N] -> [N]
        scores = scores.squeeze()  # [1,N] -> [N]

        # image-level metrics
        self.valid_metrics['AUC'] = roc_auc_score(labels, scores)
        self.valid_metrics['AP'] = average_precision_score(labels, scores)
        self.valid_metrics['PPV_at_Recall'] = ppv_at_recall(labels, scores, recall_th=0.9)
        
        """
        # others
        test_normal_score = np.mean(test_scores[np.where(test_labels == 0)])
        test_abnormal_score = np.mean(test_scores[np.where(test_labels == 1)])
        results.update({"normal_score": test_normal_score, "abnormal_score": test_abnormal_score})
        """

        # log
        d = dict()
        d['epoch'] = epoch
        d['valid_loss'] = valid_loss
        for key in self.valid_metrics.keys():
            d[f'valid_{key}'] = self.valid_metrics[key]

        self.log_dict(d, prog_bar=False, rank_zero_only=True)

        # free up the memory
        self.validation_step_outputs.clear()

        # setup flag
        self.did_validation = True

    def on_test_start(self):
        # clear GPU cache before the validation
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()        
        self.test_step_outputs.append("Video_name,Frame_id,Phase_predict")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        image = batch["image"]      # [b,c,h,w]
        video_name = batch["video_name"] 
        frame_id = batch["frame_id"] 

        # forward
        if self.ensemble == False:
            pred = self.forward(image)
            
            if self.cfg.Model.arch == 'convnext' or self.cfg.Model.arch == 'dinov2_lora': 
                pred = pred.logits
            elif self.cfg.Model.arch == 'dinov2':
    #            pred, _ = pred
                pred = pred.logits
            elif self.cfg.Model.arch == 'mstcn':
                # batch size is 1
                pred = pred[0]
        else:
            # ensemble
            preds = []
            for idx, model in enumerate(self.models):
                pred = model.forward(image.clone())
                if self.cfg.Model.arch == 'convnext' or self.cfg.Model.arch == 'dinov2_lora': 
                    pred = pred.logits
                elif self.cfg.Model.arch == 'dinov2':
                    pred = pred.logits
                elif self.cfg.Model.arch == 'mstcn':
                    # batch size is 1
                    pred = pred[0]
                preds.append(pred)
            pred = torch.stack(preds, dim=0).mean(dim=0)
            
        # for metrics 
#        print("pred(pre):", pred.shape)
        if self.cfg.Model.post_filter == True:
            pred = torch.softmax(pred, -1)
#            print("pred (before):", pred[:20,17])
            pred = post_filter(pred, filter_length=self.cfg.Model.post_filter_length)
#            print("pred (after):", pred[:20,17])
#        print("pred(post):", pred.shape)

        pred = torch.argmax(pred, -1).cpu().numpy()
        for idx, p in enumerate(pred):
            s = f"{video_name[0]},{video_name[0]}_{frame_id[idx][0].cpu().numpy()}.jpg,{p}"
            self.test_step_outputs.append(s)

    def on_test_end(self):
        # all gather
        outputs_gather = self.all_gather(self.test_step_outputs)

        if (self.cfg.Model.arch == 'convnext' or
            self.cfg.Model.arch == 'dinov2'):
            pass
            """
            if self.trainer.is_global_zero:
                # metric
                outputs = dict()
                for key in ['video_name', 'pred', 'target']:
                    outputs[key] = torch.hstack([o[key] for o in outputs_gather]).cpu().numpy()
                num_classes = self.cfg.Data.dataset.num_classes
    #            mean = calculate_metrics(outputs, num_classes)
                test_metrics = calculate_metrics_sklearn(outputs, num_classes)

                # log
                output_path = self.cfg.General.output_path
                log_filename = f"{output_path}/test_log.txt"
                with open(log_filename, "wt") as f:
                    for metric in self.test_metrics_list:
                        if metric == 'Accuracy':
                            s = f"{metric}: mean={test_metrics[metric]['mean']:.4f}, std_V={test_metrics[metric]['std_V']:.4f}"
                            print(s)
                            f.write(s + '\n')
                        else:
                            s = f"{metric}: mean={test_metrics[metric]['mean']:.4f}, std_V={test_metrics[metric]['std_V']:.4f}, std_P={test_metrics[metric]['std_P']:.4f}"
                            print(s)
                            f.write(s + '\n')
            """
        elif self.cfg.Model.arch == 'mstcn':
            # all gather
            outputs_gather = self.all_gather(self.test_step_outputs)

            if self.trainer.is_global_zero:
                output_path = self.cfg.General.output_path
                log_filename = f"{output_path}/test_results.csv"
                with open(log_filename, "wt") as f:
                    for output in outputs_gather:
                        print(output)
                        f.write(output + '\n')

    def on_predict_start(self):
        # clear GPU cache before prediction
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()

    def predict_step(self, batch, batch_idx):
        image = batch["image"]      # [b,t,c,h,w]
        case_id = batch["case_id"] 
        frame_number = batch["frame_number"]

        b, t = image.shape[:2]
        image = image.view(-1, *image.shape[2:])  # [b*t,c,h,w]

        # forward
        pred = self.forward(image)

        if self.cfg.Model.arch == 'resnet50':
            _, features = pred
            features = features.view(b, t, -1)
        elif self.cfg.Model.arch == 'dinov2':
            pred = self.forward(image, output_hidden_states=True)
            hidden_states = pred.hidden_states
            features = hidden_states[-1][:, 0, :]  # CLS token
        """
        elif self.cfg.Model.arch == 'mstcn':
            # batch size is 1
            pred = pred[0]
            target = target[0]
            video_name = [video_name[0]] * pred.shape[0]
        """    

        # to cpu
        features = features.cpu().numpy()

        # save the features to files
        for b in range(features.shape[0]):
            output_dir = f"{self.cfg.Data.outdir}/{case_id[b]}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for idx, t in enumerate(frame_number[b]):
                output_path = f"{output_dir}/{t:04d}.npy"
                np.save(output_path, features[b][idx])

    def on_predict_end(self):
        if self.trainer.is_global_zero:
            # merge npy
            topdir = self.cfg.Data.outdir
            video_dirs = sorted(glob.glob(f'{topdir}/*'))

            for video_dir in video_dirs:
                if not os.path.isdir(video_dir):
                    continue

                npy_files = sorted(glob.glob(f'{video_dir}/*.npy'))
                if len(npy_files) == 0:
                    continue

                print("video_dir:", video_dir)

                npys = []
                for npy_file in npy_files:
                    npy = np.load(npy_file)
                    npys.append(npy)
                npys = np.vstack(npys)

                output_file = f"{video_dir}.npz"
                np.savez_compressed(output_file, npys)

                # remove npy directory
                shutil.rmtree(video_dir)

        # clear the buffer
        self.predict_step_outputs.clear()

    def move_to(self, obj, device):
#        print('obj:', obj)
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.move_to(v, device)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v, device))
            return res
        elif isinstance(obj, str):
            return obj
        else:
            print('obj (error):', obj)
            raise TypeError("Invalid type for move_to")

    def configure_optimizers(self):
        conf_optim = self.cfg.Optimizer

        if hasattr(conf_optim.optimizer, 'params'):
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)
            optimizer = optimizer_cls(self.parameters(),
                                      **conf_optim.optimizer.params)
        else:
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)
            optimizer = optimizer_cls(self.parameters())

        if scheduler_cls is None:
            return {"optimizer": optimizer}
        else:
            scheduler = scheduler_cls(
                optimizer, **conf_optim.lr_scheduler.params)
            
#            print('opt, sch:', optimizer, scheduler.__class__.__name__)

            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler_config = {
                    # REQUIRED: The scheduler instance
                    "scheduler": scheduler,
                    # The unit of the scheduler's step size, could also be 'step'.
                    # 'epoch' updates the scheduler on epoch end whereas 'step'
                    # updates it after a optimizer update.
                    "interval": "epoch",
                    # How many epochs/steps should pass between calls to
                    # `scheduler.step()`. 1 corresponds to updating the learning
                    # rate after every epoch/step.
                    "frequency": 1,
                    # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                    "monitor": "valid_loss",
                    # If set to `True`, will enforce that the value specified 'monitor'
                    # is available when the scheduler is updated, thus stopping
                    # training if not found. If set to `False`, it will only produce a warning
                    "strict": True,
                    # If using the `LearningRateMonitor` callback to monitor the
                    # learning rate progress, this keyword can be used to specify
                    # a custom logged name
#                    "name": None,
                }
                return {"optimizer": optimizer,
                        "lr_scheduler": scheduler_config}
            else:             
                return {"optimizer": optimizer,
                        "lr_scheduler": scheduler}
        
    """
    def configure_optimizers(self):
        conf_optim = self.cfg.Optimizer

#        encoder_params = [p for name, p in self.named_parameters() if 'encoder' in name]
#        other_params = [p for name, p in self.named_parameters() if 'encoder' not in name]

        def is_encoder(n): return 'encoder' in n

        params = list(self.model.named_parameters())
        
        base_lr = conf_optim.optimizer.params.lr
        encoder_lr = base_lr * conf_optim.encoder_lr_ratio

        grouped_parameters = [
            {"params": [p for n, p in params if is_encoder(n)], 'lr': encoder_lr},
            {"params": [p for n, p in params if not is_encoder(n)], 'lr': base_lr},
        ]

        if hasattr(conf_optim.optimizer, 'params'):
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)
            optimizer = optimizer_cls(self.parameters(),
                                      **conf_optim.optimizer.params)
#            optimizer = optimizer_cls(grouped_parameters,
#                                      **conf_optim.optimizer.params)
        else:
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)
            optimizer = optimizer_cls(self.parameters())
#            optimizer = optimizer_cls(grouped_parameters)

        if scheduler_cls is None:
            return [optimizer]
        else:
            scheduler = scheduler_cls(
                optimizer, **conf_optim.lr_scheduler.params)
            return [optimizer], [scheduler]
    """
       
    def get_progress_bar_dict(self):
        items = dict()

        return items

    def compute_per_example_metrics(self, logits, y, gather_type="exp"):
        dtype = logits.dtype
        probs = torch.softmax(logits, -1)

        if gather_type == "exp":
            rank_output_value_array = self.rank_output_value_array.type(dtype)
            predict_y = torch.sum(probs * rank_output_value_array, dim=-1)
        elif gather_type == "max":
            predict_y = torch.argmax(probs, dim=-1).type(dtype)
        else:
            raise ValueError(f"Invalid gather_type: {gather_type}")

        y = y.type(dtype)
        mae = torch.abs(predict_y - y)
        acc = (torch.round(predict_y) == y).type(logits.dtype)

        return {f"mae_{gather_type}_metric": mae, f"acc_{gather_type}_metric": acc, "predict_y": predict_y}