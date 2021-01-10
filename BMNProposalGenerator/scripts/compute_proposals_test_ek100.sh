#mkdir -p output/ek100
#python main.py data/ek100/ models/ek100/ output/ek100/ --path_to_video_features data/ek100/video_features/ --mode inference --inference_set test
python main.py data/ek100/ models/ek100/ output/ek100/ --path_to_video_features data/ek100/video_features/ --mode postprocessing --inference_set test
