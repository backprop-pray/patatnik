#!/usr/bin/env python3
"""
Fine-tune YOLOv8n on frames collected by camera_server.py --collect.

Usage:
    python3 train.py                # train and hot-swap model
    python3 train.py --epochs 30    # more epochs
    python3 train.py --no-swap      # train without replacing yolov8n.pt
"""
import argparse
import os
import shutil
import yaml
from pathlib import Path


DATASET_DIR  = '/home/yasen/patatnik/dataset'
BASE_MODEL   = '/home/yasen/patatnik/embedded/yolov8n.pt'
RUNS_DIR     = '/home/yasen/patatnik/runs'
CLASS_NAMES  = ['plant']   # all fruit/veg/plant detections are remapped to this single class


def count_samples(dataset_dir):
    img_dir = Path(dataset_dir) / 'images'
    if not img_dir.exists():
        return 0
    return len(list(img_dir.glob('*.jpg')))


def write_data_yaml(dataset_dir):
    data = {
        'path': dataset_dir,
        'train': 'images',
        'val':   'images',   # small dataset — reuse train as val
        'nc':    len(CLASS_NAMES),
        'names': CLASS_NAMES,
    }
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',   type=int,  default=20)
    parser.add_argument('--batch',    type=int,  default=4,    help='Batch size (keep low on Pi)')
    parser.add_argument('--imgsz',    type=int,  default=640)
    parser.add_argument('--no-swap',  action='store_true',     help='Do not replace yolov8n.pt after training')
    parser.add_argument('--dataset',  default=DATASET_DIR)
    args = parser.parse_args()

    n = count_samples(args.dataset)
    if n == 0:
        print(f'No training samples found in {args.dataset}/images/')
        print('Run camera_server.py --collect first to gather data.')
        return 1

    print(f'Training on {n} samples  →  {args.epochs} epochs, batch={args.batch}')

    from ultralytics import YOLO

    yaml_path = write_data_yaml(args.dataset)
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device='cpu',
        project=RUNS_DIR,
        name='plant_finetune',
        exist_ok=True,
        verbose=False,
    )

    best = Path(results.save_dir) / 'weights' / 'best.pt'
    if not best.exists():
        print('Training finished but best.pt not found.')
        return 1

    print(f'Best weights: {best}')

    if not args.no_swap:
        backup = BASE_MODEL.replace('.pt', '_coco_backup.pt')
        if not os.path.exists(backup):
            shutil.copy(BASE_MODEL, backup)
            print(f'Backed up original model → {backup}')
        shutil.copy(str(best), BASE_MODEL)
        print(f'Model hot-swapped → {BASE_MODEL}')
        print('Restart camera_server.py to load the new weights.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
