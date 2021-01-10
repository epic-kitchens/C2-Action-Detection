#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to classify proposals."""

import argparse
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
import json
import slowfast.utils.checkpoint as cu
import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_cfg
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter, EPICTestMeter

logger = logging.get_logger(__name__)

def perform_test(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Transfer the data to the current GPU device.
        if isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            labels = labels.cuda()
        video_idx = video_idx.cuda()

        if cfg.DETECTION.ENABLE:
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

            preds = preds.cpu()
            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"].cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach().cpu(),
                ori_boxes.detach().cpu(),
                metadata.detach().cpu(),
            )
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            preds = model(inputs)

            if isinstance(labels, (dict,)):
                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    verb_preds, verb_labels, video_idx = du.all_gather(
                        [preds[0], labels['verb'], video_idx]
                    )

                    noun_preds, noun_labels, video_idx = du.all_gather(
                        [preds[1], labels['noun'], video_idx]
                    )
                    meta = du.all_gather_unaligned(meta)
                metadata = {'narration_id': []}
                for i in range(len(meta)):
                    metadata['narration_id'].extend(meta[i]['narration_id'])
                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    (verb_preds.detach().cpu(), noun_preds.detach().cpu()),
                    (verb_labels.detach().cpu(), noun_labels.detach().cpu()),
                    metadata,
                    video_idx.detach().cpu(),
                )
                test_meter.log_iter_stats(cur_iter)
            else:
                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    preds, labels, idx = du.all_gather(
                        [preds, labels, video_idx]
                    )

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    preds.detach().cpu(),
                    labels.detach().cpu(),
                    video_idx.detach().cpu(),
                )
                test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if cfg.TEST.DATASET == 'epickitchens':
        preds, labels, metadata = test_meter.finalize_metrics()
    else:
        test_meter.finalize_metrics()
        preds, labels, metadata = None, None, None
    test_meter.reset()
    return preds, labels, metadata

def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    test_list = pd.read_pickle(os.path.join(cfg.EPICKITCHENS.ANNOTATIONS_DIR, cfg.EPICKITCHENS.TEST_LIST))

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model, cfg, is_train=False)

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
            )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            )
    else:
        # raise NotImplementedError("Unknown way to load checkpoint.")
        logger.info("Testing with random initialization. Only for debugging.")

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
                len(test_loader.dataset)
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
        )
        # Create meters for multi-view testing.
        if cfg.TEST.DATASET == 'epickitchens':
            test_meter = EPICTestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
                )
        else:
            test_meter = TestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
                )

    # # Perform multi-view test on the entire dataset.
    preds, labels, metadata = perform_test(test_loader, model, test_meter, cfg)

    if du.is_master_proc():
        if cfg.TEST.DATASET == 'epickitchens':
            verb_predictions = preds[0].argmax(1)
            noun_predictions = preds[1].argmax(1)

            scores = pd.DataFrame(
                {'narration_id': metadata, 'verb': verb_predictions, 'noun': noun_predictions}).set_index(
                'narration_id')
            scores['action'] = scores['verb'].apply(str) + ',' + scores['noun'].apply(str)
            scores = scores.join(test_list)

            output_json = {
                'version': '0.2',
                'challenge': 'action_detection',
                'sls_pt': 2,
                'sls_tl': 3,
                'sls_td': 3,
                'results': {}
            }

            groups = scores.groupby('video_id')
            for vid in scores['video_id'].unique():
                annotations = groups.get_group(vid)
                detections = []
                for _, ann in annotations.iterrows():
                    detections.append({
                        'verb': ann['verb'],
                        'noun': ann['noun'],
                        'action': ann['action'],
                        'score': ann['score'],
                        'segment': [ann['start_seconds'], ann['stop_seconds']]
                    })
                output_json['results'][vid] = detections

            with open(os.path.join(cfg.OUTPUT_DIR, f"action_detection_baseline_{cfg.EPICKITCHENS.TEST_SPLIT}.json"),
                      'w') as outfile:
                json.dump(output_json, outfile)


def parse_args():
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    #cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


def main():
    """
    Main function to spawn the proposal classification process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        if cfg.NUM_GPUS > 1:
            torch.multiprocessing.spawn(
                mpu.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    test,
                    args.init_method,
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=False,
            )
        else:
            test(cfg=cfg)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()
