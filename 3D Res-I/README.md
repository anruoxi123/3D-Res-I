
Dependecies: Ubuntu 14.04, python 2.7, CUDA 8.0, cudnn 5.1, h5py (2.6.0), SimpleITK (0.10.0), numpy (1.11.3), nvidia-ml-py (7.352.0), matplotlib (2.0.0), scikit-image (0.12.3), scipy (0.18.1), pyparsing (2.1.4), pytorch (0.1.10+ac9245a) (anaconda is recommended)

Download LUNA16 dataset from https://luna16.grand-challenge.org/data/

For preprocessing, run ./DeepLung/prepare.py. The parameters for prepare.py is in config_training.py. *_data_path is the unzip raw data path for LUNA16. *_preprocess_result_path is the save path for the preprocessing. *_annos_path is the path for annotations. *_segment is the path for LUNA16 segmentation, which can be downloaded from LUNA16 website.

Use run_training.sh to train the detector. You can use 3in1 or 3in1Dropout net model by revising --model attribute. After training and test are done, use the ./evaluationScript/frocwrtdetpepchluna16.py to validate the epoch used for test. After that, collect all the 10 folds' prediction, use ./evaluationScript/noduleCADEvaluationLUNA16.py to get the FROC for all 10 folds. You can directly run noduleCADEvaluationLUNA16.py, and get the performance in the paper.

The performances on each fold are follows:
3D Res-I/evaluationScript/annotations/3D Res-I.csv




