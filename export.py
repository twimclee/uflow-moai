import torch
import torch.onnx
import onnx

import argparse

from src.model import UFlow

from pathfilemgr import MPathFileManager
from hyp_data import MHyp, MData




p = argparse.ArgumentParser()
p.add_argument('--volume', help='volume directory', default='moai')
p.add_argument('--project', help='project directory', default='test_project')
p.add_argument('--subproject', help='subproject directory', default='test_subproject')
p.add_argument('--task', help='task directory', default='test_task')
p.add_argument('--version', help='version', default='v1')
args, _ = p.parse_known_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("loading parameters")

mpfm = MPathFileManager(args.volume, args.project, args.subproject, args.task, args.version)
mhyp = MHyp()
mpfm.load_test_hyp(mhyp)

print("loading model")

model = UFlow(mhyp.input_size, mhyp.flow_steps, mhyp.backbone)
model.from_pretrained(f'{mpfm.weight_path}/best.ckpt')
model.eval()
model = model.to(DEVICE)

input_tensor = torch.randn(1, 3, mhyp.input_size, mhyp.input_size).to(DEVICE)
onnx_path = f'{mpfm.weight_path}/best.onnx'


print("start exporting...")

torch.onnx.export(model,                    # 모델
                input_tensor,               # 입력 텐서
                onnx_path,                  # 저장할 ONNX 파일 경로
                export_params=True,         # 가중치 내보내기
                opset_version=16,           # ONNX 버전
                input_names=['input'],      # 입력 텐서 이름
                output_names=['output'],    # 출력 텐서 이름
                dynamic_axes=None)          # 배치 크기 동적 설정

print("done")