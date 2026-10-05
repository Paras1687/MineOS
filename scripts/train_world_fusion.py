import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.config import ARTIFACTS
from backend.models.geo_intelligence.world_fusion import WorldFusionCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--epochs', type=int, default=16)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with zipfile.ZipFile(args.bundle) as archive:
        data = np.load(archive.open('world_unknown_training/arrays.npz'))
        images = np.asarray(data['x'], dtype=np.float32)
        gravity = np.asarray(data['g'], dtype=np.float32).reshape(-1, 1)
        labels = np.asarray(data['y'], dtype=np.float32)
        groups = np.asarray(data['groups'])
        model_source = archive.read('world_unknown_training/result.json')
        gravity_source = archive.read('gravity/world_bouguer.grd')
    if len(images) != 1782 or int(labels.sum()) != 446 or int((labels == 0).sum()) != 1336:
        raise ValueError(f'Unexpected archive data: {len(images)} samples, {int(labels.sum())} positives')
    if len(groups) != len(labels) or images.shape[1:] != (64, 64, 9):
        raise ValueError('Prepared sample shape or group manifest mismatch')
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    grid_path = ARTIFACTS / 'world_bouguer.grd'
    grid_path.write_bytes(gravity_source)
    mean, variance = float(gravity.mean()), float(gravity.var())
    model = WorldFusionCNN(mean, variance)
    class_weights = torch.tensor([len(labels)/(2*(labels == c).sum()) for c in (0, 1)], dtype=torch.float32)
    x = torch.from_numpy(images.transpose(0, 3, 1, 2).copy())
    g, y = torch.from_numpy(gravity), torch.from_numpy(labels)
    loader = DataLoader(TensorDataset(x, g, y), batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    started = time.time()
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for bx, bg, by in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx, bg)
            weights = class_weights[by.long()]
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, by, reduction='none') * weights).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(by)
        print(f'epoch {epoch+1}/{args.epochs} loss={total/len(y):.5f}', flush=True)
    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(model(x, g)).numpy()
    state_path = ARTIFACTS / 'world_fusion.pt'
    torch.save(model.state_dict(), state_path)
    metadata = dict(version='moil-world-fusion-full-v1', architecture='PyTorch port of bundled 3-block CNN + Bouguer fusion',
        training_samples=len(labels), positive_samples=int(labels.sum()), unknown_background_samples=int((labels == 0).sum()),
        unique_spatial_groups=len(set(groups.tolist())), epochs=args.epochs, seed=42, gravity_mean=mean,
        gravity_variance=variance, training_loss=None, full_dataset_fit=True, calibrated=False,
        source_split_metrics_reference=json.loads(model_source),
        warning='All supplied positive and Negative_Unknowns samples were used in fitting. In-sample scores are not independent validation. Unknown-background labels do not establish barren ground or mineralization.',
        training_seconds=round(time.time()-started, 1), score_summary=dict(min=float(scores.min()),max=float(scores.max()),mean=float(scores.mean())))
    metadata['training_loss'] = float(total/len(y))
    metadata['model_sha256'] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    metadata['gravity_sha256'] = hashlib.sha256(grid_path.read_bytes()).hexdigest()
    (ARTIFACTS/'world_fusion_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ['version','training_samples','positive_samples','unknown_background_samples','epochs','training_loss','training_seconds']}), flush=True)


if __name__ == '__main__':
    main()
