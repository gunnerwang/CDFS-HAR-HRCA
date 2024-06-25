# CDFS-HAR-HRCA
This repository contains the codes and some sample data for the paper "Data-efficient multimodal human action recognition for proactive human–robot collaborative assembly: A cross-domain few-shot learning approach".

## Dataset
- **Some Sample Data of Our Collection**:
https://drive.google.com/file/d/1k5luBFxdE6hcmopY8kbSIY-QtIydzRIE/view?usp=sharing (three input modalities of RGB, Depth, Skeleton)
- **CoAx**: https://dlgmtzs.github.io/dataset-coax
- **HRI30**: https://zenodo.org/records/5833411
- **UCF101**: https://www.crcv.ucf.edu/data/UCF101.php

Each downloaded dataset should be placed in ```video_datasets/data/<dataset>```.

## Usage
### Pretraining
We provide the pretrained model weights at ```checkpoint_dir_coax```, ```checkpoint_dir_coax_depth```, ```checkpoint_dir_hri30``` and ```checkpoint_dir_ucf55```. If you would like to pretrain the model from scratch or based on these checkpoints for other datasets, you can run the command below:
```
bash scripts/pretrain_<dataset>.sh
```

### Inference
The model inference proceeds in a few-shot manner along with the cross-domain test-time adaptation. 
```
bash scripts/inference.sh
```

## Acknowledgement
Our method is related to the projects [TRX](https://github.com/tobyperrett/trx) and [URL](https://github.com/VICO-UoE/URL) in the computer vision field.

## Citation
If you find our work helpful for your research, please consider citing the following BibTeX entry.
```
@article{wang2024data,
  title={Data-efficient multimodal human action recognition for proactive human--robot collaborative assembly: A cross-domain few-shot learning approach},
  author={Wang, Tianyu and Liu, Zhihao and Wang, Lihui and Li, Mian and Wang, Xi Vincent},
  journal={Robotics and Computer-Integrated Manufacturing},
  volume={89},
  pages={102785},
  year={2024},
  publisher={Elsevier}
}
```
