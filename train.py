"""
python src/train.py -cat carpet -config configs/carpet.yaml -train_dir ../training/carpet/
"""

import warnings
import argparse
from pathlib import Path

import torch
from ignite.contrib import metrics
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from torchvision.utils import make_grid
from torchvision import transforms

from src.miou import mIoU
from src.aupro import AUPRO
from src.model import UFlow
from src.nfa_tree import compute_nfa_anomaly_score_tree
from src.datamodule import UFlowDatamodule, uflow_un_normalize, get_debug_images_paths
from src.callbacks import MyPrintingCallback, ModelCheckpointByAuROC, ModelCheckpointByAuPRO, ModelCheckpointBymIoU
from src.callbacks import ModelCheckpointByInterval

import csv
from pathfilemgr import MPathFileManager
from hyp_data import MHyp, MData
from utiles import TimeManager, CSVManager
from test import predict


# warnings.filterwarnings("ignore", category=UserWarning, message="Your val_dataloader has `shuffle=True`")
# warnings.filterwarnings("ignore", category=UserWarning, message="Checkpoint directory .* exists and is not empty")

LOG = 0


torch.set_float32_matmul_precision('medium')

class UFlowTrainer(LightningModule):

    def __init__(
        self,
        args,
        flow_model,
        datamodule,
        learning_rate=1e-3,
        weight_decay=1e-7,
        log_every_n_epochs=25,
        save_debug_images_every=25,
        log_predefined_debug_images=True,
        log_n_images=20,
        val_path=None,
        epochs=0,
        csv_mgr=None,
        time_mgr=None
    ):
        """

        @param flow_model:
        @param learning_rate:
        @param weight_decay:
        @param log_every_n_epochs:
        @param save_debug_images_every:
        @param log_n_images:
        """
        super(UFlowTrainer, self).__init__()
        self.args = args
        self.model = flow_model
        self.datamodule = datamodule
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.log_every_n_epochs = log_every_n_epochs
        self.save_debug_images_every = save_debug_images_every
        self.log_predefined_debug_images = log_predefined_debug_images
        self.log_n_images = log_n_images
        self.debug_img_size = 256
        self.debug_img_resizer = transforms.Compose([transforms.Resize(self.debug_img_size)])
        self.debug_images = get_debug_images_paths(val_path)

        # Metrics
        self.pixel_auroc = metrics.ROC_AUC()
        self.pixel_aupro = AUPRO()
        self.image_auroc = metrics.ROC_AUC()
        self.miou_lnfa0 = mIoU(thresholds=[0])

        # Debug images
        self.test_images = None
        self.test_targets = None

        self.pixel_auroc_val = 0.0
        self.pixel_aupro_val = 0.0
        self.image_auroc_val = 0.0
        # self.miou_val = 0.0

        self.epochs = epochs
        self.csv_mgr = csv_mgr
        self.time_mgr = time_mgr

    def step(self, batch, batch_idx):
        z, ljd = self.model(batch)

        # Compute loss
        lpz = torch.sum(torch.stack([0.5 * torch.sum(z_i ** 2, dim=(1, 2, 3)) for z_i in z]), dim=0)
        flow_loss = torch.mean(lpz - ljd)

        return {"01_Train/Loss": flow_loss.detach(), "loss": flow_loss}

    def training_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self.log_dict({"loss": losses['loss']}, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return losses
 
    def on_train_epoch_end(self):
        if self.logger is not None:
            # Get the learning rate
            def get_lr(optimizer):
                for param_group in optimizer.param_groups:
                    return param_group['lr']
            self.logger.experiment.add_scalar("04_LearningRate", get_lr(self.optimizers()), self.current_epoch)


            remaining = self.time_mgr.get_time_left(self.current_epoch, self.epochs)
            print(f"Writing to CSV: epoch={self.current_epoch}, remaining={remaining}")
            self.csv_mgr.writerow([
                self.current_epoch, 
                remaining,
                self.pixel_auroc_val,
                self.pixel_aupro_val,
                self.image_auroc_val,
                # self.miou_val
            ])

    def validation_step(self, batch, batch_idx):
        images, targets, paths = batch

        # if self.current_epoch == 0:
        #     debug_images = self.debug_images
        #     # Keep predefined images to show in all different trainings always the same ones
        #     if self.log_predefined_debug_images and (len(debug_images) > 0):
        #         if batch_idx == 0:
        #             self.test_images = torch.Tensor(len(debug_images), *images.shape[1:]).to(images.device)
        #             self.test_targets = torch.Tensor(len(debug_images), *targets.shape[1:]).to(targets.device)
        #         for i, img_path in enumerate(paths):
        #             try:
        #                 idx = [i for i, s in enumerate(debug_images) if s in img_path][0]
        #                 self.test_images[idx] = images[i]
        #                 self.test_targets[idx] = targets[i]
        #             except IndexError:
        #                 pass
        #     # Or randomly sample a different set for each training
        #     else:
        #         if batch_idx == 0:
        #             self.test_images = torch.Tensor().to(images.device)
        #             self.test_targets = torch.Tensor().to(targets.device)
        #         n_to_keep = self.log_n_images - batch_idx * images.shape[0]
        #         if n_to_keep > 0:
        #             self.test_images = torch.cat([self.test_images, images[:n_to_keep, ...]], dim=0)
        #             self.test_targets = torch.cat([self.test_targets, targets[:n_to_keep, ...]], dim=0)

        # Update metrics
        if self.current_epoch % self.log_every_n_epochs == 0:
            images, targets, paths = batch
            z, _ = self.model(images)
            z = [zz.detach() for zz in z]

            # Pixel level metrics
            anomaly_score = 1 - self.model.get_probability(z, self.debug_img_size)
            # lnfa = compute_nfa_anomaly_score_tree(z, self.debug_img_size)
            resized_targets = 1 * (self.debug_img_resizer(targets) > 0.5)
            self.pixel_auroc.update((anomaly_score.ravel(), resized_targets.ravel()))
            self.pixel_aupro.update(anomaly_score, resized_targets)
            # self.miou_lnfa0.update(lnfa.detach().cpu(), resized_targets.cpu())

            # Image level metric
            image_targets = torch.IntTensor([0 if 'good' in p else 1 for p in paths])
            image_anomaly_score = torch.amax(anomaly_score, dim=(1, 2, 3))
            self.image_auroc.update((image_anomaly_score.ravel().cpu(), image_targets.ravel().cpu()))

        # predict(self.args, self.model, self.datamodule)

    def on_validation_epoch_end(self) -> None:
        # Log metrics
        if self.current_epoch % self.log_every_n_epochs == 0:
            # Compute metrics
            pixel_auroc = float(self.pixel_auroc.compute())
            pixel_aupro = float(self.pixel_aupro.compute())
            image_auroc = float(self.image_auroc.compute())
            # pixel_miou = float(self.miou_lnfa0.compute().numpy())
            self.log_dict(
                {'pixel_auroc': pixel_auroc, 'pixel_aupro': pixel_aupro, 'image_auroc': image_auroc},
                on_step=False, on_epoch=True, prog_bar=False, logger=True
            )

            self.pixel_auroc_val = pixel_auroc
            self.pixel_aupro_val = pixel_aupro
            self.image_auroc_val = image_auroc
            # self.miou_val = pixel_miou

            self.pixel_auroc.reset()
            self.pixel_aupro.reset()
            self.image_auroc.reset()
            # self.miou_lnfa0.reset()

            self.logger.experiment.add_scalar("03_ValidationMetrics/PixelAuROC", pixel_auroc, self.current_epoch)
            self.logger.experiment.add_scalar("03_ValidationMetrics/PixelAuPRO", pixel_aupro, self.current_epoch)
            self.logger.experiment.add_scalar("03_ValidationMetrics/ImageAuROC", image_auroc, self.current_epoch)
            # self.logger.experiment.add_scalar("03_ValidationMetrics/PixelmIoU", pixel_miou, self.current_epoch)

        # Log example images
        # if self.current_epoch % self.save_debug_images_every == 0:
        #     anomaly_maps_to_show = []
        #     for i in range(len(self.test_images)):
        #         # Forward
        #         batch = self.test_images[i:i+1, ...]
        #         outputs, _ = self.model(batch)
        #         # Compute probabilities
        #         anomaly_maps_to_show.append(1 - self.model.get_probability(outputs, self.debug_img_size))

        #     # Generate output probability images
        #     images_grid = make_grid(
        #         uflow_un_normalize(self.debug_img_resizer(self.test_images)).to('cpu'),
        #         normalize=True, nrow=1, value_range=(0, 1)
        #     )
        #     labels_grid = make_grid(
        #         self.debug_img_resizer(self.test_targets).to('cpu'),
        #         normalize=True, nrow=1, value_range=(0, 1)
        #     )
        #     anomaly_maps_grid = make_grid(
        #         torch.cat(anomaly_maps_to_show, dim=0).to('cpu'),
        #         normalize=True, nrow=1, value_range=(0, 1)
        #     )
        #     to_show = torch.dstack((images_grid, labels_grid, anomaly_maps_grid))
        #     self.logger.experiment.add_image(
        #         f"Example images",
        #         to_show,
        #         self.current_epoch,
        #         dataformats="CHW"
        #     )
        
        # predict(self.args, self.model, self.datamodule)
        # current_epoch = self.current_epoch
        # if (current_epoch + 1) % 10 == 0:
        #     predict(self.args, self.model, self.datamodule)

    def configure_optimizers(self):
        def get_total_number_of_iterations():
            try:
                self.trainer.reset_train_dataloader()
                number_of_training_examples = len(self.trainer.train_dataloader.dataset)
                batch_size = self.trainer.train_dataloader.loaders.batch_size
                drop_last = 1 * self.trainer.train_dataloader.loaders.drop_last
                iterations_per_epoch = number_of_training_examples // batch_size + 1 - drop_last
                total_iterations = iterations_per_epoch * (self.trainer.max_epochs - 1)
            except:
                total_iterations = 25000
            return total_iterations

        # Optimizer
        optimizer = torch.optim.Adam(
            [{"params": self.parameters(), "initial_lr": self.lr}],
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Scheduler for slowly reducing learning rate
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1., end_factor=0.4, total_iters=get_total_number_of_iterations()
        )
        return [optimizer], [scheduler]

def train(args):
    mpfm = MPathFileManager(args.volume, args.project, args.subproject, args.task, args.version)
    mhyp = MHyp()
    mpfm.load_train_hyp(mhyp)
    mpfm.save_hyp(mhyp)

    # Data
    # ------------------------------------------------------------------------------------------------------------------
    input_size = mhyp.input_size
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    image_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=mhyp.brightness, contrast=mhyp.contrast, saturation=mhyp.saturation, hue=mhyp.hue),
            transforms.RandomHorizontalFlip(p=mhyp.hflip),
            transforms.RandomVerticalFlip(p=mhyp.vflip),
            transforms.RandomRotation(degrees=mhyp.rotation),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )

    datamodule = UFlowDatamodule(
        data_dir=mpfm.train_dataset,
        input_size=input_size,
        batch_train=mhyp.batch_train,
        batch_test=mhyp.batch_val,
        image_transform=image_transform,
        workers=mhyp.workers,
        shuffle_test=True,
        mode='train'
    )

    valimage_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )

    valdatamodule = UFlowDatamodule(
        data_dir=mpfm.val_path, # 데이터셋 경로로 validation 이 들어감
        input_size=input_size,
        batch_train=1,
        batch_test=10,
        image_transform=valimage_transform,
        shuffle_test=False,
        mode='valtest'
    )

    ###### result.csv
    time_mgr = TimeManager()
    print(f"Creating CSV file at: {mpfm.result_csv}")
    csv_mgr = CSVManager(mpfm.result_csv, ["epoch", "time", "pixel_auroc_val", "pixel_aupro_val", "image_auroc_val"])

    # Model
    # ------------------------------------------------------------------------------------------------------------------
    uflow = UFlow(mhyp.input_size, mhyp.flow_steps, mhyp.backbone)
    # uflow = torch.compile(uflow)
    uflow_trainer = UFlowTrainer(
        args,
        uflow,
        valdatamodule,
        mhyp.learning_rate,
        mhyp.weight_decay,
        mhyp.log_every_n_epochs,
        mhyp.save_debug_images_every,
        mhyp.log_predefined_debug_images,
        mhyp.log_n_images,
        mpfm.val_path,
        mhyp.epochs,
        csv_mgr,
        time_mgr
    )



    # Train
    # ------------------------------------------------------------------------------------------------------------------
    callbacks = [
        MyPrintingCallback(),
        # ModelCheckpointByAuROC(mpfm.weight_path),
        # ModelCheckpointByAuPRO(mpfm.weight_path),
        # ModelCheckpointBymIoU(mpfm.weight_path),
        ModelCheckpointByInterval(mpfm.weight_path, mhyp.save_ckpt_every),
        LearningRateMonitor('epoch'),
        EarlyStopping(
            monitor="pixel_auroc",
            mode="max",
            patience=mhyp.patience,
        ),
    ]
    logger = TensorBoardLogger(save_dir=mpfm.train_result, name=None, version=None)

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=mhyp.epochs + 1,
        log_every_n_steps=10,
        callbacks=callbacks,
        logger=logger,
        # check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        default_root_dir=mpfm.train_result
    )

    time_mgr.start()

    print("Starting training...")
    try:
        trainer.fit(uflow_trainer, 
            train_dataloaders=datamodule.train_dataloader(), 
            val_dataloaders=datamodule.val_dataloader())
    finally:
        print("Training finished or interrupted. Closing CSV file...")
        csv_mgr.close()
        print("CSV file closed.")

if __name__ == "__main__":
    # seed_everything(0)

    # Args
    # ------------------------------------------------------------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument('--volume', help='volume directory', default='moai')
    p.add_argument('--project', help='project directory', default='20250115')
    p.add_argument('--subproject', help='subproject directory', default='test_sub')
    p.add_argument('--task', help='task directory', default='test_uflow')
    p.add_argument('--version', help='version', default='v1')
    cmd_args, _ = p.parse_known_args()

    # Execute
    # ------------------------------------------------------------------------------------------------------------------
    train(cmd_args)
