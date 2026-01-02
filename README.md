
# High Dimensionality Visualization

## Repo structure

```
exp0.ipynb                  # notebook for experiment 0
exp1.ipynb                  # notebook for experiment 1
exp2.ipynb                  # notebook for experiment 2
exp3.ipynb                  # notebook for experiment 3
exp4.ipynb                  # notebook for experiment 4
exp5.ipynb                  # notebook for experiment 5
README.md
```

## Experiments

### Exp 0:
Raw image flattening and UMAP visualization on MNIST and Imagenette datasets.

### Exp 1:
Multi-layer neural network trained on synthetic spiral data with latent space visualization across network depth.

### Exp 2:
ResNet18 trained on MNIST comparing scratch vs. pretrained (ImageNet) initialization.

### Exp 3:
ResNet50 trained on Imagenette with progressive UMAP projections and error analysis during training.

### Exp 4:
Testing UMAP generalization: training on different sample sizes and projecting test data onto learned embeddings.

### Exp 5:
MNIST-trained ResNet18 projected onto KMNIST to visualize out-of-distribution behavior in latent space.

## Datasets used

- **MNIST**: 28×28 grayscale digit images (10 classes)
- **KMNIST**: Kuzushiji characters (10 classes, Japanese historical data)
- **Imagenette (v2)**: 10-class image subset of ImageNet
- **Toy spiral**: 3-class 2D spiral dataset

## Dependencies

- ```torch```
- ```torchvision```
- ```umap-learn```
- ```matplotlib```
- ```numpy```

## Usage

Download/clone the repo and run each notebook entirely to execute the associated experiment. All of them are independent and can be executed separately. Each experiment handles missing data by downloading it automatically when needed and storing it in a folder named ```data``` by default in your working directory.

## Additional notes

- Performances can vary from one run to another (especially on experiment 1), the results we show in our report and presentation have been obtained using the parameters already present when you initially download the repo (no need to change anything).
- Imports are systematically separated in two cells, this has been done to avoid a kernel crash on our end but it is not necessary to keep it that way if it works for you.
