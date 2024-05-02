from EmbedSeg.utils.create_dicts import create_test_configs_dict
from EmbedSeg.test import begin_evaluating

def main():
    configs = create_test_configs_dict(
        data_dir = r"C:/Users/cryst/Desktop/Test Images",
        checkpoint_path = r"C:/Users/cryst/Desktop/Organoid-Segmentation-Web-App/models/best_iou_model.pth",
        tta = True,
        ap_val = 0.5,
        min_object_size = 2,
        save_dir = r"C:/Users/cryst/Desktop/Organoid-Segmentation-Web-App/temp",
        norm = 'min-max-percentile',
        data_type = '8-bit',
        one_hot = False,
        n_y = 1392,
        n_x = 1392,
        cluster_fast = True,
        device = "cpu"
    )



    result = begin_evaluating(configs)

if __name__ == '__main__':
    main()