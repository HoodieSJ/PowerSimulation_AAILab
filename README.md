# Score Matching Model for Unbounded Data Score

This code provides an pytorch implementation for the paper `Ordered Risk and Confidence Regularization for Robust Training from Biased Dataset`.
Our paper introduces a debiasing algorithm, coined Ordered Risk and Confidence regularization (ORC), that relatively regularizes the confidence and the risk of the subgroups in the dataset.

--------------------
## How to run the code

### Dependencies

The following dependencies are required to run the code.
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

Train vanilla, V-REX (baseline), and ORC (our model) through `main.sh`

```sh
main.py:
  --config: python file path for the training configuration
    (default: 'None')
  --assetdir: The folder name of assets for the stats file
  --eval_folder: The folder name for storing evaluation results such as samples
    (default: 'eval')
  --mode: <train|eval> eval mode contains likelihood evaluation and FID/IS computation
  --workdir: Working directory to save all artifacts such as checkpoints/samples/log
```
## Results
Out model achieves the following performance on:

### Image Generation on CIFAR10

| Experimentsal Setup | NLL (BPD) | FID-50k | IS-50k |
|:----------|:-------:|:----------:|:----------:|
| `cifar10_uncsn_1e-3/` | 2.96 | 2.55 | 9.97 |
| `cifar10_uncsn_deep_1e-3_mid/` | 2.83 | **2.33** | **10.11** |
| `cifar10_uncsn_deep_1e-5_mid/` | **2.06** | 2.58 | 9.74 |
| `cifar10_uncsn_deep_1e-5/` | 2.35 | 2.38 | 9.87 |
