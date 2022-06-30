# Simple and Effective Synthesis of Indoor 3D Scenes

This repository hosts the open source code for [SE3DS](https://arxiv.org/abs/2204.02960).

[![Video Results](./video_results.gif)]({https://www.youtube.com/watch?v=lhwwlrRfFp0} "Video Results")

[Paper](https://arxiv.org/abs/2204.02960) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()


## Setup instructions

### Environment
Set up virtualenv, and install required libraries:
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add the SE3DS library to PYTHONPATH:
```
export PYTHONPATH=$PYTHONPATH:/home/path/to/se3ds_root/
```

### Downloading Pretrained Checkpoints

We provide the pretrained checkpoint (trained on Matterport3D) used in our paper for reporting generation and VLN results. The checkpoint can be downloaded by running:
```
wget https://storage.googleapis.com/gresearch/se3ds/mp3d_ckpt.tar -P data/
tar -xf data/mp3d_ckpt.tar --directory data/
```

The results will be extracted to the `data/` directory. The checkpoint is quite large (17GB), so make sure there is sufficient disk space.



## Colab Inference Demo

### Matterport3D

`notebooks/SE3DS_MP3D_Example_Colab.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() shows how to setup and run the pretrained SE3DS model (trained on Matterport3D trajectories) for inference. It includes examples on synthesizing image sequences and continuous video sequences for arbitrary navigation trajectories in Matterport3D.

### RealEstate10K

`notebooks/SE3DS_MP3D_Example_Colab.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() shows how to setup and run the pretrained SE3DS model (trained on RE10K videos) for inference.


## VLN Perturbation Augmentation

`notebooks/SE3DS_VLN_Augmentation_Colab.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() shows how to run the pretrained SE3DS model for the VLN augmentation experiment we have in the paper. In our experiments, this improves the success rate of an already strong VLN agent by up to 1.5%.


## Training a New Model

We provide example training and testing scripts (`train.sh` and `test.sh`). After preprocessing data in the expected TFRecords format (see `datasets/indoor_datasets.py` and `datasets/indoor_datasets_test.py`), update the gin config files to specify the location of the tfrecords for train (`R2RImageDataset.data_dir`) and eval (`R2RVideoDataset.data_dir`).

### Training

Start a training run, by first editing train.sh to specify an appropriate work directory. By default, the script uses all GPUs available, and distributes data across all available GPUs. After configuring the training job, start an experiment by running it on bash:

```
mkdir exp
bash train.sh exp_name &> train.txt
```

### Evaluation

To run an evaluation job, update test.sh with the correct settings used in the training script. Then, execute

```
bash test.sh exp_name &> eval.txt
```

to start an evaluation job. All checkpoints in the work directory will be evaluated for FID and Inception Score. If you can spare the GPUs, you can also run train.sh and test.sh in parallel, which will continuously evaluate new checkpoints saved into the work directory. Scores will be written to Tensorboard and output to eval.txt.


## Citation

If you find this work useful, please consider citing:

```
@article{koh2022simple,
  title={Simple and Effective Synthesis of Indoor 3D Scenes},
  author={Koh, Jing Yu and Agrawal, Harsh and Batra, Dhruv and Tucker, Richard and Waters, Austin and Lee, Honglak and Yang, Yinfei and Baldridge, Jason and Anderson, Peter},
  journal={arXiv preprint arXiv:2204.02960},
  year={2022}
}
```

## License

SE3DS is released under the Apache 2.0 license. The Matterport3D dataset is governed by the
[Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).

## Disclaimer

This is not an officially supported Google product.
