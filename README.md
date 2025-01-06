# EthicalFab (Ethical Fabrication, SME NAMRC 2024)

## 1. Environment configuration

The code is based on Python 3.12+ and requires CUDA version 11.0 or higher. Follow the specific configuration steps below to set up the environment:

1.  Create a conda environment:
   
   ```shell
   conda create -y -n ethicalfab python=3.12
   conda activate ethicalfab
   ```

2. Install PyTorch (version can vary depending on your environment):
   
   ```shell
   conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
   ```

## 2. Dataset
You can download dataset in this [link](http://bit.ly/2YOEa5Z).

 ```shell
@inproceedings{ma2020database,
  title={Database and benchmark for early-stage malicious activity detection in 3D printing},
  author={Ma, Xiaolong and Li, Zhe and Li, Hongjia and An, Qiyuan and Qiu, Qinru and Xu, Wenyao and Wang, Yanzhi},
  booktitle={2020 25th Asia and South Pacific design automation conference (ASP-DAC)},
  pages={494--499},
  year={2020},
  organization={IEEE}
}
```


## 3. Run by ipynb code
Each file have the result of different model with differential privacy (DP) or non-DP.
The result will be saved in 'save' folder as .pkl file.

We would like to suggest to run this code in GPU condition.
