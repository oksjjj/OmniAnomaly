# -*- coding: utf-8 -*-
"""
Download SMAP / MSL dataset (telemanom) into ./data/

The original S3 URL (s3-us-west-2.amazonaws.com/telemanom/data.zip) returns 403.
This script downloads from Hugging Face: appleparan/telemanom
"""
import os
import shutil
import sys

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = 'appleparan/telemanom'
DATA_DIR = 'data'


def download_smap_msl():
    os.makedirs(os.path.join(DATA_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'test'), exist_ok=True)

    print('Downloading labeled_anomalies.csv ...')
    csv_path = hf_hub_download(
        repo_id=REPO_ID,
        filename='labeled_anomalies.csv',
        repo_type='dataset',
    )
    shutil.copy2(csv_path, os.path.join(DATA_DIR, 'labeled_anomalies.csv'))

    npy_files = [
        f for f in list_repo_files(REPO_ID, repo_type='dataset')
        if f.startswith('data/data/train/') and f.endswith('.npy')
        or f.startswith('data/data/test/') and f.endswith('.npy')
    ]
    print(f'Downloading {len(npy_files)} .npy files ...')

    for i, remote_path in enumerate(sorted(npy_files), 1):
        category = 'train' if '/train/' in remote_path else 'test'
        filename = os.path.basename(remote_path)
        local_path = os.path.join(DATA_DIR, category, filename)

        if os.path.exists(local_path):
            print(f'[{i}/{len(npy_files)}] skip {category}/{filename}')
            continue

        cached = hf_hub_download(repo_id=REPO_ID, filename=remote_path, repo_type='dataset')
        shutil.copy2(cached, local_path)
        print(f'[{i}/{len(npy_files)}] {category}/{filename}')

    print(f'\nDone. Dataset ready at ./{DATA_DIR}/')
    print('Next: python data_preprocess.py SMAP')
    print('      python data_preprocess.py MSL')


if __name__ == '__main__':
    try:
        download_smap_msl()
    except Exception as e:
        print('Download failed:', e, file=sys.stderr)
        print('\nAlternative (Kaggle API key required):', file=sys.stderr)
        print('  pip install kaggle', file=sys.stderr)
        print('  kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl', file=sys.stderr)
        sys.exit(1)
