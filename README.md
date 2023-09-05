# A wavelet-aware Lightweight Hybrid Model for Fast Low-Light Enhancement

## Abstract
Low-light enhancement task is an essential component of computer low-level visual tasks, which involves processing images captured under dim lighting conditions to make them appear as if they were taken under normal illumination.  In order to solve the above problems, we propose a lightweight baseline that combines CNN and sparse grid attention transformer blocks to enable the model to capture a global receptive field at an early stage. Specifically, we propose a High-Frequency Wavelet-aware Block(HFWB) that focuses on processing high-frequency information in the wavelet domain to refine details and suppress noise. With a processing time of only 10.6ms, the performance of our model outperforms that of the current state-of-the-art lightweight models on benchmark lowlight datasets. Compared to state-of-the-art models in the LOL dataset, our model achieves a reduction in inference time of over 90% and requires only about 1% of the FLOPS.

This repository contains the dataset, code, and pre-trained models for our paper.


## Dataset
You can use the following links to download the datasets.The LOL v1 dataet has 485 pairs of images for training and 15 pairs images for testing. The MIT-FiveK dataset has 4500 images for training ans 498 images for testing. 

- [LOL v1]  [[Link]](https://daooshee.github.io/BMVC2018website/)

- [MIT-Adobe-FiveK] [[Link]](https://github.com/HuiZeng/Image-Adaptive-3DLUT)

## Pretrained Model

We provide our pre-trained models in the pre folder

## Get Started
### train & evaluation

1.We provide our shell training file, you can use the shell to train& evaluation.

The train_v1.sh is used to  train the LOL v1 dataset, and the train_mit.sh corresponds to the Mit-FiveK dataset.For example ,you can use the following command to train lol v1 dateset :

```sh
$ sh ./train_v1.sh
```

2.Also,We provide our shell testing file, you can use following command to evaluate pretrained model .

```sh
$ sh ./test_v1.sh
```

### shell arguments

The shell files mentioned above have some options to config , such as:

```sh
       --images_path           "/path/to/your/train/data" \
       --images_val_path       "/path/to/your/val/data" \
       --snapshots_folder     "/output/STAR-DCE" \
       --out_folder          "/output/"  \
       --snapshots_folder     /output/STAR-DCE \
       --num_epochs 300 \
       --mir_n_feat  16 \
       --mir_chan_factor  2 \
       --train_batch_size  8 \
       --max_lr      1e-3 \
       --max_weight_decay  0.005 \
```

you should modify the path to make it suitable for your environment.




