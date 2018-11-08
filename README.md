# FOTS: Fast Oriented Text Spotting with a Unified Network

### Introduction
This is a pytorch re-implementation of [FOTS: Fast Oriented Text Spotting with a Unified Network](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1699.pdf).
The features are summarized blow:

+ Only **detection** part is implemented.

### Contents
1. [Installation](#installation)
2. [Download](#download)
3. [Train](#train)
4. [Test](#test)


### Installation
1. Any version of torch version >= 0.3.1 should be ok.

### Download
1. Models trained on ICDAR 2015 (training set) + ICDAR 2017 (training set)

### Train
If you want to train the model, you should provide the dataset path, in the dataset path, a separate gt text file should be provided for each image
and run

```
python main_train.py

```

### Test
run
```
python eval.py
```

a text file will be then written to the output path.

