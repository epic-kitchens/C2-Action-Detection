import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        'path_to_dataset',
        type=str
    )

    parser.add_argument(
        'checkpoint_path',
        type=str
    )

    parser.add_argument(
        'output_path',
        type=str
    )

    parser.add_argument(
        '--rgb_lmdb',
        type=str
    )

    parser.add_argument(
        '--flow_lmdb',
        type=str
    )

    parser.add_argument(
        '--path_to_video_features',
        type=str,
        default='video_features'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='train'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30
    )

    parser.add_argument(
        '--sigma',
        type=int,
        default=15
    )

    parser.add_argument(
        '--fname_template',
        type=str,
        default='frame_{:010d}.jpg'
    )

    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.001
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4
    )

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=9
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16
    )

    parser.add_argument(
        '--step_size',
        type=int,
        default=7
    )

    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--observation_window',
        type=int,
        default=400
    )

    parser.add_argument(
        '--max_duration',
        type=int,
        default=400
    )

    parser.add_argument(
        '--num_sample',
        type=int,
        default=4
    )

    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3
    )

    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5
    )

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=2048
    )

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=32
    )

    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.4
    )

    parser.add_argument(
        '--soft_nms_low_thres',
        type=float,
        default=0.5
    )

    parser.add_argument(
        '--post_processing_topk',
        type=int,
        default=1000
    )

    parser.add_argument(
        '--max_proposals',
        type=int,
        default=1000
    )

    parser.add_argument(
        '--ppm',
        type = int,
        default = 200
    )

    parser.add_argument(
        '--soft_nms_high_thres',
        type=float,
        default=0.9
    )

    parser.add_argument(
        '--result_file',
        type=str,
        default="/result_proposal.pkl"
    )

    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="/evaluation_result.jpg"
    )

    parser.add_argument(
        '--inference_set',
        type=str,
        default="validation"
    )

    args = parser.parse_args()

    return args

