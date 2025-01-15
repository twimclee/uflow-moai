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

from pathfilemgr import MPathFileManager
from hyp_data import MHyp, MData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TARGET_SIZE = 256

def predict(args, trained=None):
    mpfm = MPathFileManager(args.volume, args.project, args.subproject, args.task, args.version)
    mhyp = MHyp()
    mpfm.load_test_hyp(mhyp)

    input_size = mhyp.input_size
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    image_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )


    data_path = None
    if trained is None:
        data_path = mpfm.test_dataset
    else:
        data_path = f"{mpfm.val_path}/good"

    # Data
    datamodule = UFlowDatamodule(
        data_dir=data_path,
        input_size=input_size,
        batch_train=1,
        batch_test=10,
        image_transform=image_transform,
        shuffle_test=False,
        is_train=False
    )

    progress_bar = tqdm(datamodule.test_dataloader())
    # progress_bar.set_description(f"Test")

    # Load model
    # flow_model = UFlow(**config['model'])
    # flow_model.from_pretrained(Path("models") / "auc" / f"{args.category}.ckpt")

    model = None
    save_path = None
    if trained is None:
        flow_model = UFlow(mhyp.input_size, mhyp.flow_steps, mhyp.backbone)
        flow_model.from_pretrained(f'{mpfm.weight_path}/best.ckpt')
        flow_model.eval()
        model = flow_model.to(DEVICE)
        save_path = mpfm.test_result
    else:
        model = trained
        save_path = mpfm.evaluate_result

    model.eval()
    all_images, all_targets, all_scores, all_lnfas = [], [], [], []
    for images, targets, img_paths in progress_bar:
        print(img_paths)
        with torch.no_grad():
            z, _ = model.forward(images.to(DEVICE))

        all_scores.append(1 - model.get_probability(z, input_size))
        all_lnfas.append(compute_nfa_anomaly_score_tree(z, input_size))
        all_images.append(np.clip(uflow_un_normalize(
            F.interpolate(images, [input_size, input_size], mode="bilinear", align_corners=False)), 0, 1))
        all_targets.append(F.interpolate(targets, [input_size, input_size], mode="bilinear", align_corners=False))

    all_scores = torch.cat(all_scores, dim=0)
    all_lnfas = torch.cat(all_lnfas, dim=0)
    all_images = torch.cat(all_images, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    score_min, score_max = np.percentile(all_scores.cpu(), 1.), np.percentile(all_scores.cpu(), 99.)
    lnfa_min, lnfa_max = np.percentile(all_lnfas, 1.), np.percentile(all_lnfas, 99.)

    # Ensure the result folder exists
    # result_dir = Path("result") / args.category
    # result_dir.mkdir(parents=True, exist_ok=True)

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
        plt.savefig(f"{save_path}/likelihood_{idx}_{score.mean().item():.4f}.png", bbox_inches='tight')
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
        plt.savefig(f"{save_path}/log_nfa_{idx}_{score.mean().item():.4f}.png", bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Args
    # ------------------------------------------------------------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument('--volume', help='volume directory', default='moai')
    p.add_argument('--project', help='project directory', default='test_project')
    p.add_argument('--subproject', help='subproject directory', default='test_subproject')
    p.add_argument('--task', help='task directory', default='test_task')
    p.add_argument('--version', help='version', default='v1')
    cmd_args, _ = p.parse_known_args()

    # Execute
    # ------------------------------------------------------------------------------------------------------------------
    predict(cmd_args)
