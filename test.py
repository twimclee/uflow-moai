import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
import yaml
from tqdm import tqdm

from src.nfa_tree import compute_nfa_anomaly_score_tree
from src.datamodule import UFlowDatamodule, uflow_un_normalize
from src.model import UFlow

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = 256


def predict(args):

    config = yaml.safe_load(open(Path("configs") / f"{args.category}.yaml", "r"))

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    image_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )

    # Data
    datamodule = UFlowDatamodule(
        data_dir=args.data,
        category=args.category,
        input_size=config['model']['input_size'],
        batch_train=1,
        batch_test=10,
        image_transform=image_transform,
        shuffle_test=False
    )

    progress_bar = tqdm(datamodule.val_dataloader())
    progress_bar.set_description(f"{args.category.upper()}")

    # Load model
    flow_model = UFlow(**config['model'])
    flow_model.from_pretrained(Path("models") / "auc" / f"{args.category}.ckpt")
    flow_model.eval()
    model = flow_model.to(DEVICE)

    all_images, all_targets, all_scores, all_lnfas = [], [], [], []
    for images, targets, img_paths in progress_bar:
        with torch.no_grad():
            z, _ = model.forward(images.to(DEVICE))

        all_scores.append(1 - model.get_probability(z, TARGET_SIZE))
        all_lnfas.append(compute_nfa_anomaly_score_tree(z, TARGET_SIZE))
        all_images.append(np.clip(mvtec_un_normalize(
            F.interpolate(images, [TARGET_SIZE, TARGET_SIZE], mode="bilinear", align_corners=False)), 0, 1))
        all_targets.append(F.interpolate(targets, [TARGET_SIZE, TARGET_SIZE], mode="bilinear", align_corners=False))

    all_scores = torch.cat(all_scores, dim=0)
    all_lnfas = torch.cat(all_lnfas, dim=0)
    all_images = torch.cat(all_images, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    score_min, score_max = np.percentile(all_scores.cpu(), 1.), np.percentile(all_scores.cpu(), 99.)
    lnfa_min, lnfa_max = np.percentile(all_lnfas, 1.), np.percentile(all_lnfas, 99.)

    # Ensure the result folder exists
    result_dir = Path("result") / args.category
    result_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img, target, score, lnfa) in enumerate(zip(all_images, all_targets, all_scores, all_lnfas)):
        # Likelihood heatmap
        plt.figure(1)
        plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
        heatmap = np.clip((score[0].detach().cpu().numpy() - score_min) / (score_max - score_min), 0, 1)
        plt.imshow(heatmap, alpha=0.4, cmap='turbo', vmin=0, vmax=1)
        if target.sum() > 0:
            plt.contour(target[0].detach().cpu().numpy(), [0.5])
        plt.title('Likelihood')
        plt.axis('off')
        plt.tight_layout()

        # Score 파일 저장
        plt.savefig(result_dir / f"likelihood_{idx}_{score.mean().item():.4f}.png", bbox_inches='tight')
        plt.close()

        # Log(NFA) heatmap
        plt.figure(2)
        plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
        heatmap = np.clip((lnfa[0].detach().cpu().numpy() - lnfa_min) / (lnfa_max - lnfa_min), 0, 1)
        plt.imshow(heatmap, alpha=0.4, cmap='turbo', vmin=0, vmax=1)
        if target.sum() > 0:
            plt.contour(target[0].detach().cpu().numpy(), [0.5])
        plt.title('Log(NFA)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(result_dir / f"log_nfa_{idx}_{score.mean().item():.4f}.png", bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Args
    # ------------------------------------------------------------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("-cat", "--category", default="carpet", type=str)
    p.add_argument("-data", "--data", default="data/mvtec", type=str)
    cmd_args, _ = p.parse_known_args()

    # Execute
    # ------------------------------------------------------------------------------------------------------------------
    predict(cmd_args)
