from lightning.pytorch.callbacks import Callback, ModelCheckpoint

class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("\nTraining is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


class ModelCheckpointByAuROC(ModelCheckpoint):
    def __init__(self, experiment_path):
        super(ModelCheckpointByAuROC, self).__init__(
            monitor='pixel_auroc',
            dirpath=str(experiment_path),
            mode='max',
            # filename='best_val_roc_auc__epoch_{epoch:04d}__pixel_auroc_{pixel_auroc:.4f}',
            # filename='best_pixel_auroc',
            filename='best',
            auto_insert_metric_name=False,
        )

class ModelCheckpointByAuPRO(ModelCheckpoint):
    def __init__(self, experiment_path):
        super(ModelCheckpointByAuPRO, self).__init__(
            monitor='pixel_aupro',
            dirpath=str(experiment_path),
            mode='max',
            # filename='best_val_roc_pro__epoch_{epoch:04d}__pixel_aupro_{pixel_aupro:.4f}',
            filename='best_pixel_aupro',
            auto_insert_metric_name=False,
        )

class ModelCheckpointBymIoU(ModelCheckpoint):
    def __init__(self, experiment_path):
        super(ModelCheckpointBymIoU, self).__init__(
            monitor='miou',
            dirpath=str(experiment_path),
            mode='max',
            # filename='best_val_miou_nfa__epoch_{epoch:04d}__miou_{miou:.4f}',
            filename='best_nfa_miou',
            auto_insert_metric_name=False,
        )

class ModelCheckpointByInterval(ModelCheckpoint):
    def __init__(self, experiment_path, every_n_epochs=50):
        super(ModelCheckpointByInterval, self).__init__(
            dirpath=str(experiment_path),
            every_n_epochs=every_n_epochs,
            save_top_k=-1,
            filename='epoch_{epoch:04d}__train_loss_{loss:.4f}__pixel_auroc_{pixel_auroc:.4f}',
            auto_insert_metric_name=False,
            save_on_train_epoch_end=False
        )
