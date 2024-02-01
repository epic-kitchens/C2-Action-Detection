# C2-Action-Detection

## Challenge
To participate and submit to this challenge, register at the [Action Detection Codalab Challenge](https://codalab.lisn.upsaclay.fr/competitions/707).

## Evaluation Code
This repository contains the official code to evaluate egocentric action detection methods on the EPIC-KITCHENS-100 validation set. Parts of the evaluation code have been adapted from https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_detection.py. 

Note that on 01/02/2024 the evaluatiuon code has been updated to fix some minor errors. As a result re-evaluating the same submission may lead to minor changes in performance evaluation.

To use this code, move to the `EvaluationCode` directory:

```
cd EvaluationCode
```

### Requirements
In order to use the evaluation code, you will need to install a few packages. You can install these requirements with: 

```
pip install -r requirements.txt
```

### Usage
You can use this evaluation code to evaluate submissions on the validation set in the official JSON format. To do so, you will need to first download the public EPIC-KITCHENS-100 annotations with:

```
export PATH_TO_ANNOTATIONS=/desired/path/to/annotations
git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git $PATH_TO_ANNOTATIONS
```

You can then evaluate your json file with:

```
python evaluate_detection_json_ek100.py /path/to/json $PATH_TO_ANNOTATIONS/EPIC_100_validation.pkl
```

Where `/path/to/json` is the path to the json file to be evaluated and `/path/to/annotations` is the path to the cloned `epic-kitchens-100` repository.

### Example json file
As an example, we provide a json file generated with the baseline on the validation set. You can evaluate the json file with:

```
python evaluate_detection_json_ek100.py action_detection_baseline_validation.json $PATH_TO_ANNOTATIONS/EPIC_100_validation.pkl
```

## Action Detection Baseline
The action detection baseline used for the experiments is composed of two modules:
 * The action proposal generator, based on [Boundary Matching Networks](https://arxiv.org/abs/1907.09702);
 * An action proposal classifier, which transforms the action proposals into detections. We use SlowFast networks trained on EPIC-KITCHENS-100;
 
In the following, we provide instructions to train/test each of these modules.

### Action proposals generator
We used the PyTorch implementation available at https://github.com/JJBOY/BMN-Boundary-Matching-Network. We modified the original implementation to allow training and testing on EPIC-KITCHENS-100. We provide our modified version in the `BMN` directory. Before continuing, change directory with:

```
cd BMNProposalGenerator
```

#### Requirements
We recommend to use [Anaconda](http://anaconda.org/). You can install requirements with:

```
conda env create -f environment.yml
```

Then you can activate the newly created environment with:

```
conda activate c2-action-detection-bmn
```

#### Features
We provide video features re-scaled to an observation window equal to 400 to facilitate training and test of the models using the default parameters considered in our [EPIC-KITCHENS-100 paper](https://arxiv.org/abs/2006.13256). You can download the features with:

```
chmod +x scripts/download_data_ek100.sh
./scripts/download_data_ek100.sh
```

If you want to change the observation window size during training, you should download the full set of features, which will be resized to the desired observation window on the fly:

```
chmod +x scripts/download_data_ek100_full.sh
./scripts/download_data_ek100_full.sh
```

#### Model
You can download the model used to report baseline results in the paper with:

```
chmod +x scripts/download_models_ek100.sh
./scripts/download_models_ek100.sh
```

#### Validation and Test
We provide generated validation and test action proposals in the `output/ek100` directory. These proposals can be generated with the following scripts:

```
chmod +x scripts/compute_proposals_validation_ek100.sh
./scripts/compute_proposals_validation_ek100.sh
```

```
chmod +x scripts/compute_proposals_test_ek100.sh
./scripts/compute_proposals_test_ek100.sh
```

Note that this process may require some time. The scripts will use the model for inference and generate two detection files containing the object proposals:

 * `output/ek100/result_proposal-validation.pkl`
 * `output/ek100/result_proposal-test.pkl`
 
#### Training
 You can train the models with:

```
chmod +x scripts/train_ek100.sh
./scripts/train_ek100.sh
```

Mind that the parameters have been tuned to train the model on 4 V100 (16 GB) GPUS, hence a large amount of GPU memory is required for training.

##### Training with a non-default observation window
If you want to train with a non-default observation window, you first have to download the full set of features. Then, you have to train the model specifying the locations of the rgb and flow lmdb files:

```
python main.py \
    data/ek100/ \
    models/ek100/bmn/ \
    output/ek100/ \
    --path_to_video_features data/ek100/video_features/ 
    --rgb_lmdb data/ek100/rgb \
    --flow_lmdb data/ek100/flow \
    --observation_window window_size
```

Note that the if `data/ek100/video_features` is not empty, the code will try to load features form there before loading the lmdb datasets. It could be a good idea to empty the `video_features` directory before training if training parameters have been chenged.

### Action Proposals Classifier
We provide code to classify action proposals which is based on https://github.com/epic-kitchens/epic-kitchens-slowfast. To use the action proposal classifier, you will need to first install the epic-kitchens-slowfast codebase following the instructions available at https://github.com/epic-kitchens/epic-kitchens-slowfast. After installing, you should add the repository to your PYTHONPATH with:

```
export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH
```

Then enter the `SlowFastProposalClassifier` directory with:

```
cd SlowFastProposalClassifier
```

#### RGB Frames
Please follow the instructions at https://github.com/epic-kitchens/epic-kitchens-slowfast#preparation to download all rgb frames.

#### Model
Please follow the instructions at https://github.com/epic-kitchens/epic-kitchens-slowfast#pretrained-model to obtain the pre-trained SlowFast model.

#### Compute Validation Detections
You can compute the validation detections using the following command:

```
python run_net.py --cfg /path/to/epic-kitchens-slowfast/configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml \
  NUM_GPUS num_gpu \
  OUTPUT_DIR .. \
  EPICKITCHENS.VISUAL_DATA_DIR /path/to/frames \
  EPICKITCHENS.ANNOTATIONS_DIR /path/to/epic-kitchens-100-annotations \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATH /path/to/slowfast-checkpoint.pyth \
  EPICKITCHENS.TEST_LIST result_proposal-validation.pkl \
  EPICKITCHENS.TEST_SPLIT validation \
  TEST.BATCH_SIZE batch_size 
```

where:
 * `/path/to/epic-kitchens-slowfast` is the path to the `epic-kitchens-slowfast` installation (https://github.com/epic-kitchens/epic-kitchens-slowfast);
 * `/path/to/frames` is the path to the directory containing all frames. Please see `https://github.com/epic-kitchens/epic-kitchens-slowfast#preparation`;
 * `/path/to/epic-kitchens-100-annotations` is the path to the cloned `epic-kitchens-100-annotations` repository (https://github.com/epic-kitchens/epic-kitchens-100-annotations);
 * `/path/to/slowfast-checkpoint.pyth` is the path to the SlowFast model checkpoint. See https://github.com/epic-kitchens/epic-kitchens-slowfast#pretrained-model;
 
 After running the command a json containing the validation detections will be generated. you can evaluate it with:
 
```
cd ..
python evaluate_detection_json_ek100.py  action_detection_baseline_validation.json path_to_annotations
```

#### Compute Test Detections
You can compute the test detections using the following command:

```
python run_net.py --cfg /path/to/epic-kitchens-slowfast/configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml \
  NUM_GPUS num_gpu \
  OUTPUT_DIR .. \
  EPICKITCHENS.VISUAL_DATA_DIR /path/to/frames \
  EPICKITCHENS.ANNOTATIONS_DIR /path/to/epic-kitchens-100-annotations \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATH /path/to/slowfast-checkpoint.pyth \
  EPICKITCHENS.TEST_LIST result_proposal-test.pkl \
  EPICKITCHENS.TEST_SPLIT test \
  TEST.BATCH_SIZE batch_size 
```
 
 After running the command a json containing the test detections will be generated. you can evaluate it with:
 
```
cd ..
python evaluate_detection_json_ek100.py  action_detection_baseline_test.json path_to_annotations
```

#### Training of the SlowFast model
Please follow the instructions at https://github.com/epic-kitchens/epic-kitchens-slowfast#trainingvalidation for information on how to train the SlowFast model.
