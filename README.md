# Ordered Risk and Confidence Regularization for Robust Training from Biased Dataset

This code provides an pytorch implementation for the paper `Ordered Risk and Confidence Regularization for Robust Training from Biased Dataset`.
Our paper introduces a debiasing algorithm, coined Ordered Risk and Confidence regularization (ORC), that relatively regularizes the confidence and the risk of the subgroups in the dataset. 

We provide experiments on Colored-MNIST and Corrupted CIFAR-10 datasets with various skewedness and random seeds.

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
      - Sometimes, HTTP error occurs. Please try a few times more, when HTTP error occurs.

2. Train vanilla, V-REX (baseline), and ORC (our model) through provided bash files.

      - train vanilla model with Colored-MNIST : `train_mnist_vanilla.sh`
      - train V-REX model with Colored-MNIST : `train_mnist_rex.sh`
      - train ORC model with Colored-MNIST : `train_mnist_orc.sh`

      - train vanilla model with Corrupted-CIFAR10 : `train_cifar10_vanilla.sh`
      - train V-REX model with Corrupted-CIFAR10 : `train_cifar10_rex.sh`
      - train ORC model with Corrupted-CIFAR10 : `train_cifar10_orc.sh`

## Results

Python file prints the test accuracy (whole accuracy, bias-aligned accuracy, bias-skewed accuracy) per validation steps.

The results of each experiment is saved on following directory with pickle file format as follows:

For the result of vanilla model with Colored-MNIST Skewed0.05 at random seed 0:
'log/colored_mnist/result/ColoredMNIST-Skewed0.05-Severity1_var5/final_result_o.pickle'

For the result of V-REX model with Colored-MNIST Skewed0.05 at random seed 0:
'log/colored_mnist/result/ColoredMNIST-Skewed0.05-Severity1_var5_REX/final_result_o.pickle'

For the result of ORC model with Colored-MNIST Skewed0.05 at random seed 0:
'log/colored_mnist/result/ColoredMNIST-Skewed0.05-Severity1_var5_sm0.15_ORC/final_result_o.pickle'

For the result of vanilla model with Corrupted CIFAR-10 Skewed0.05 at random seed 0:
'log/corrupted_cifar/result/CorruptedCIFAR10-Type0-Skewed0.05-Severity1_var5/final_result_o.pickle'

For the result of V-REX model with Colored-MNIST Skewed0.05 at random seed 0:
'log/corrupted_cifar/result/CorruptedCIFAR10-Type0-Skewed0.05-Severity1_var5_REX/final_result_o.pickle'

For the result of ORC model with Colored-MNIST Skewed0.05 at random seed 0:
'log/corrupted_cifar/result/CorruptedCIFAR10-Type0-Skewed0.05-Severity1_var5_sm0.15_ORC/final_result_o.pickle'

