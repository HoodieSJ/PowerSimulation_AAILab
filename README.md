# Ordered Risk and Confidence Regularization for Robust Training from Biased Dataset

This code provides an pytorch implementation for the paper `Ordered Risk and Confidence Regularization for Robust Training from Biased Dataset`.
Our paper introduces a debiasing algorithm, coined Ordered Risk and Confidence regularization (ORC), that relatively regularizes the confidence and the risk of the subgroups in the dataset.

--------------------
## How to run the code

### Dependencies

The following dependencies are required to run the code on Python3.6 version:
```
numpy==1.18.2
opencv-python==4.2.0.32
torch==1.7.0+cu110
Pillow==7.1.1
sacred==0.8.1
scikit-image==0.15.0
scikit-learn==0.21.3
scipy==1.3.0
tqdm==4.35.0
Wand==0.5.8
pandas
seaborn
```
### Usage

1. First, install Colored-MNIST & Corrupted CIFAR-10 : `install_for_python36.sh`

2. Train vanilla, V-REX (baseline), and ORC (our model) through provided bash files.

      - train vanilla model with Colored-MNIST : `train_mnist_vanilla.sh`
      - train V-REX model with Colored-MNIST : `train_mnist_rex.sh`
      - train ORC model with Colored-MNIST : `train_mnist_orc.sh`

      - train vanilla model with Corrupted-CIFAR10 : `train_cifar10_vanilla.sh`
      - train V-REX model with Corrupted-CIFAR10 : `train_cifar10_rex.sh`
      - train ORC model with Corrupted-CIFAR10 : `train_cifar10_orc.sh`


## Results
Out model achieves the following performance on:


