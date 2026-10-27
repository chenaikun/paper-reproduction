from pathlib import Path

import torch
from mmengine.config import Config

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'gfl_r18_voc_baseline.py'


def reduce_loss(value):
    if isinstance(value, list):
        return sum(item.mean() for item in value)
    return value.mean()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this check.')

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(CONFIG))
    model = MODELS.build(cfg.model).cuda().train()
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    sample = dataset[0]
    batch = model.data_preprocessor(
        {
            'inputs': [sample['inputs']],
            'data_samples': [sample['data_samples']],
        },
        training=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.optim_wrapper.optimizer.lr,
        momentum=0.9,
        weight_decay=0.0001,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        losses = model(
            inputs=batch['inputs'],
            data_samples=batch['data_samples'],
            mode='loss',
        )
        total_loss = sum(reduce_loss(value) for value in losses.values())
    total_loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    print(f'dataset_size: {len(dataset)}')
    print(f'input_shape: {tuple(batch["inputs"].shape)}')
    print(f'losses: { {key: float(reduce_loss(value).detach()) for key, value in losses.items()} }')
    print(f'total_loss: {float(total_loss.detach()):.6f}')
    print(f'peak_memory_mib: {torch.cuda.max_memory_allocated() / 1024**2:.1f}')


if __name__ == '__main__':
    main()
