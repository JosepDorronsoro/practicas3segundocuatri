# -*- coding: utf-8 -*-
"""
Diffusion Model — Custom Cat Dataset (64x64) — VP-Coseno
Entrenamiento en una sola fase con lr fijo:
  · lr=1e-4, 3000 épocas  (desde cero)

Ejecutar desde la carpeta raíz del proyecto:
    python train_cats_cosine.py
"""

import sys
import os
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)
for path_entry in (PROJECT_DIR, PROJECT_DIR.parent):
    if str(path_entry) not in sys.path:
        sys.path.append(str(path_entry))

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision.transforms import ToTensor
from PIL import Image
import tqdm

from diffusion_lib import (
    VPProcess,
    CosineSchedule,
    EulerMaruyamaSampler,
    GenerativeDiffusionModel,
    UNetScoreModelColor,
)

# ── Config ────────────────────────────────────────────────────────────────────
CAT_FOLDER = 'otras_pruebas/cat_AI_v2/cats'
BATCH_SIZE = 256
N_EPOCHS   = 3000
LR         = 1e-4

# T=0.999 obligatorio con CosineSchedule: beta(t) → ∞ cuando t → 1
T_COSINE = 0.999

# Checkpoints en escala logarítmica: ~15 épocas distribuidas de log(1) a log(3000),
# más el epoch final garantizado.
_log_ckpts = np.unique(
    np.round(np.logspace(0, np.log10(N_EPOCHS), 15)).astype(int)
)
LOG_CHECKPOINTS = set(_log_ckpts.tolist()) | {N_EPOCHS}

CKPT_DIR = Path('checkpoints_cats_cosine')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
print(f'Schedule: VP-Coseno  (T={T_COSINE})')
print(f'lr={LR:.0e}  |  {N_EPOCHS} épocas')
print(f'Épocas con checkpoint: {sorted(LOG_CHECKPOINTS)}')


# ── Dataset ───────────────────────────────────────────────────────────────────
class CatFolderDataset(Dataset):
    def __init__(self, folder_path: str):
        transform = ToTensor()
        paths = sorted(Path(folder_path).glob('*.jpg'))
        if not paths:
            raise FileNotFoundError(
                f'No se encontraron imágenes .jpg en {folder_path}'
            )
        print(f'Cargando {len(paths)} imágenes en RAM...')
        self.images = [transform(Image.open(p).convert('RGB')) for p in paths]
        print(f'Dataset listo. Shape: {self.images[0].shape}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], 0


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Dataset
    dataset    = CatFolderDataset(CAT_FOLDER)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == 'cuda'),
    )

    # Proceso VP-Coseno
    vp = VPProcess(schedule=CosineSchedule(), T=T_COSINE)

    # Score model
    score_model = UNetScoreModelColor(
        marginal_prob_std=vp.sigma_t
    ).to(DEVICE)

    diffusion_model = GenerativeDiffusionModel(
        vp, EulerMaruyamaSampler(), score_model, DEVICE
    )
    optimizer = Adam(score_model.parameters(), lr=LR)

    print(f'\n{"="*60}')
    print(f' VP-Coseno  |  lr={LR:.0e}  |  {N_EPOCHS} épocas')
    print(f'{"="*60}')

    tqdm_epoch = tqdm.trange(1, N_EPOCHS + 1, desc='Entrenando')
    last_ckpt  = None

    for epoch in tqdm_epoch:
        avg_loss  = 0.0
        num_items = 0

        for x, _ in dataloader:
            x = x.to(DEVICE)
            loss = diffusion_model.compute_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss  += loss.item() * x.shape[0]
            num_items += x.shape[0]

        epoch_loss = avg_loss / num_items
        tqdm_epoch.set_description(f'loss={epoch_loss:.5f}')

        if epoch in LOG_CHECKPOINTS:
            fname     = f'cat64_VP-Cos_lr{LR:.0e}_ep{epoch:04d}.pth'
            ckpt_path = CKPT_DIR / fname
            torch.save(score_model.state_dict(), ckpt_path)
            tqdm_epoch.write(f'  [ckpt] época {epoch:4d} → {fname}')
            last_ckpt = ckpt_path

    print(f'\nEntrenamiento completo. Último checkpoint: {last_ckpt}')
