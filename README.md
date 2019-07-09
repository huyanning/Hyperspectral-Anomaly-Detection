# Hyperspectral-Anomaly-Detection 
Paper and Code about my research on hyperpsectral anomaly detection 

### Introduction

This python code is about our work on hyperspectral anomaly detection named *"Hyperspectral Anomaly Detection via Background and Potential Anomaly Detection"* in https://ieeexplore.ieee.org/abstract/document/8519775. 

The main technique in this work includes **Joint Sparse Representation** and **Low Rank and Sparse Representation**. Please see the paper for more details.

### Citing my paper:
If you find this work useful in your research, please consider citing:

    @article{huyan2018hyperspectral,
    title={Hyperspectral Anomaly Detection via Background and Potential Anomaly Dictionaries Construction},
    author={Huyan, Ning and Zhang, Xiangrong and Zhou, Huiyu and Jiao, Licheng},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    volume={57},
    number={4},
    pages={2263--2276},
    year={2018},
    publisher={IEEE}
    }


### Contents
1. [Requirements](#Requirements)
2. [Example Dataset](#Example-Dataset)
3. [Code detail](#Code-detail)

### Requirements
- Python3 enviroment with Numpy and Scipy.
- Available in both Windows and Linux.
Please note that, the way importing dataset is different in different environment!
- The code has been teseted in Windows 10 Intel(R) Core(TM) i7-7700 CPU

### Example Dataset
The dataset was provided by the author of *"Anomaly detection in hyperspectral images based on low-rank and sparse representation"*. It was collected by the Airborne Visible/Infrared Imaging Spectrometer over San Diego, CA, USA (AVIRIS). The spatial resolution is 3.5m per pixel. It has 224 spectral bands in the wavelengths ranging from 370 to 2510 nm. If you want to use that you can also cite

    @article{xu2015anomaly,
    title={Anomaly detection in hyperspectral images based on low-rank and sparse representation},
    author={Xu, Yang and Wu, Zebin and Li, Jun and Plaza, Antonio and Wei, Zhihui},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    volume={54},
    number={4},
    pages={1990--2000},
    year={2015},
    publisher={IEEE}
    }


### Code detail
