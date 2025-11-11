# EthicalFab (Ethical Fabrication, SME NAMRC 2024)

## 1. Environment configuration

The code is based on Python 3.12+ and requires CUDA version 11.0 or higher. Follow the specific configuration steps below to set up the environment:

1.  Create a conda environment:
   
   ```shell
   conda create -y -n ethicalfab python=3.12
   conda activate ethicalfab
   ```

2. Install requirement PackagesPyTorch (version can vary depending on your environment):

   install torch based on your condition in this [link](https://pytorch.org/get-started/locally/).
   
```shell
python -m pip install "opencv-python==4.9.0.80"
pip install Pillow
python -m pip install opacus --no-deps
pip uninstall numpy
pip install numpy==1.26.0  # numpy version must be lower than 2.0.0 for using opacus
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
## 3. Run 'changing_file_structure.ipynb' file

This file changes file structure

#### The original file structure

 ```shell
./
├── lstm_images/
│   ├── Cup
│   ├── FathersDay
│   ├── ...
```

#### when you run the first line

```shell
./
├── train/
│   ├── Cup
│   ├── FathersDay
│   ├── ...
├── test/
│   ├── Cup
│   ├── FathersDay
│   ├── ...
```

#### when you run the second line

```shell
./
├── train_split/
│   ├── Cup
│   │   ├── CupR0P0Y0_0
│   │   ├── CupR0P0Y0_1
│   │   ├── ...
│   ├── FathersDay
│   │   ├── FathersDayR0P0Y0_0
│   │   ├── FathersDayR0P0Y0_1
│   │   ├── ...

│   ├── ...
├── test_split/
│   ├── Cup
│   │   ├── CupR5P5Y5_2
│   │   ├── CupR5P5Y5_1
│   │   ├── ...
│   ├── FathersDay
│   │   ├── FathersDayR4P3Y2_1
│   │   ├── FathersDayR4P3Y2_2
│   │   ├── ...
```


Use generated train_split and test_split as the path to run the ipynb file in the below.


## 4. Run by ipynb code
Each file have the result of different model with differential privacy (DP) or non-DP.
The result will be saved in 'save' folder as .pkl file.

Change epoch and batch size based on your situation. We set epoch and batch size as 20 and 512. 

We would like to suggest to run this code in GPU condition.

## 5. Citation
If you use this repository in your research, please cite:
```
@article{kang2025ethicalfab,
  title={EthicalFab: Toward ethical fabrication process through privacy-preserving illegal product detection},
  author={Kang, Minsung and Sun, Hongyue},
  journal={Manufacturing Letters},
  volume={44},
  pages={1425--1431},
  year={2025},
  publisher={Elsevier}
}
```
