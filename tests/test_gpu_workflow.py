"""Cache identity, physical bone geometry and batched-prior integration regressions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from anatoskel.data import ScanDataset, canonical_from_mask, generate_synthetic_dataset, read_manifest
from anatoskel.data.scans import bone_voxels_world
from anatoskel.training.checkpoint import load_checkpoint
from anatoskel.training.pipeline import train_model, train_prior
from scripts.prepare_training_cache import prepare_one


def test_tight_geometry_preserves_world_centroid_and_pca(tmp_path):
    scan=ScanDataset(generate_synthetic_dataset(tmp_path))[0]
    mask, affine, world=bone_voxels_world(scan,20)
    native=(scan['labels']==20)
    full=canonical_from_mask(native,scan['affine'])
    tight=canonical_from_mask(mask,affine)
    np.testing.assert_allclose(tight['translation'],full['translation'],atol=1e-8)
    np.testing.assert_allclose(tight['rotation'],full['rotation'],atol=1e-8)
    np.testing.assert_allclose(world.mean(0),full['translation'],atol=1e-8)
    assert mask.numel()<native.numel()
    assert scan['labels'].dtype==torch.uint8


def test_cached_physical_scans_batch_prior_and_threaded_main_training(tmp_path):
    torch.set_num_threads(2)
    manifest=generate_synthetic_dataset(tmp_path/'raw')
    settings=yaml.safe_load((Path(__file__).parents[1]/'configs/smoke.yaml').read_text())
    cache=tmp_path/'cache'
    cache.mkdir()
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    record=read_manifest(manifest)[0]
    prepare_one((record,str(cache),digest,settings['data'],1))
    (cache/'cache.json').write_text(json.dumps({'status':'complete','data_config':settings['data']}))
    scan=ScanDataset(manifest,'train',cache_dir=cache)[0]
    assert scan['volume'].dtype==torch.float32
    assert 'cached_coarse' in scan
    settings['train'].update(cache_dir=str(cache),prior_batch_size=2,roi_workers=2,
                              fine_label_ids=[20,21])
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(settings))
    prior=train_prior(config,manifest,tmp_path/'prior',device='cpu',epochs=1)
    payload=load_checkpoint(prior['checkpoint'])
    assert payload['prior_samples_seen']==len(payload['canonical_frames'])==2
    assert (payload['shape_codes'].norm(dim=1)>0).all()
    result=train_model(config,manifest,tmp_path/'train',device='cpu',
                         prior_path=prior['checkpoint'],max_steps=1)
    assert result['last']['fine_bones']==2
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError,match='different manifest'):
        ScanDataset(manifest,'train',cache_dir=cache)[0]
