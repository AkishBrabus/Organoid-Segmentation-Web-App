import os
from EmbedSeg.utils.create_dicts import create_test_configs_dict
from EmbedSeg.test import begin_evaluating

def segment(
        data_dir, 
        save_dir, 
        checkpoint_path, 
        tta=False, ap_val=0.5, 
        min_object_size=2,
        norm = 'min-max-percentile',
        data_type = '8-bit',
        one_hot = False,
        n_y = 1392,
        n_x = 1392,
        cluster_fast = True,
        device = 'cpu',
        progress_bar = None
        ):
    
    os.path.join("temp", "comp", "uploaded")
    
    configs = create_test_configs_dict(
        data_dir = data_dir,
        checkpoint_path = checkpoint_path,
        tta = tta,
        ap_val = ap_val,
        min_object_size = min_object_size,
        save_dir = save_dir,
        norm = norm,
        data_type = data_type,
        one_hot = one_hot,
        n_y = n_y,
        n_x = n_x,
        cluster_fast = cluster_fast,
        device = device
    )

    result = begin_evaluating(configs, progress_bar=progress_bar)