# DSE Final Project - InfoGAN Disentanglement Enhancement

This repository explores disentanglement techniques for InfoGAN (Information Maximizing Generative Adversarial Network) models. The implementation focuses on enhancing the model's ability to learn interpretable and separated latent factors through various regularization methods.

## Overview

InfoGAN is an extension of traditional GANs that learns disentangled representations in a completely unsupervised manner by maximizing mutual information between a subset of latent variables and the observed data. This project implements and compares four variants:

- **InfoGAN**: The baseline implementation as described in the original paper
- **InfoGAN-OR**: InfoGAN with Orthogonal Regularization which encourages independent latent factors
- **InfoGAN-CR**: InfoGAN with Contrastive Regularization which promotes consistent latent space traversal
- **InfoGAN-ORCR**: InfoGAN with both Orthogonal and Contrastive Regularization combined

The code trains all four models on the MNIST dataset and provides comprehensive metric comparisons to evaluate disentanglement quality.

## Results

Our experiments show that:

1. Orthogonal regularization enhances factor independence significantly
2. Contrastive regularization improves traversal linearity and semantic control
3. Combined regularization achieves the best overall disentanglement metrics

Each model is evaluated using multiple metrics including categorical accuracy, continuous correlation, factor independence, traversal linearity, and mutual information.

## Instructions to Reproduce Results

### Using Google Colab

1. Open the notebook in Google Colab: [Open in Colab](https://colab.research.google.com/)
2. Upload the `InfoGAN_ORCR_Implementation.ipynb` Jupyter notebook to the Colab environment
3. Run the notebook with a T4 GPU runtime:
   - Go to Runtime → Change runtime type
   - Select "GPU" as Hardware accelerator
4. Execute the code cell by cell or run the entire script at once

### Automatic Installation in Colab
python!pip install -q torch torchvision matplotlib seaborn pandas scikit-learn tqdm

### Configuration Options
The script includes several configuration options that you can modify:
- `FAST_MODE`: Set to True for quicker training/evaluation, False for comprehensive metrics
- `N_EPOCHS`: Number of training epochs (default: 10)
- `ORTHOGONAL_WEIGHT_REG`: Weight for orthogonal regularization (default: 0.1)
- `CR_WEIGHT`: Weight for contrastive regularization (default: 0.2)

### Memory Requirements

The models are trained sequentially to optimize memory usage
Recommended: At least 12GB GPU memory (T4 or better)
RAM: 8GB minimum, 16GB recommended

### Viewing TensorBoard Visualizations
After training, you can view detailed metrics and visualizations in TensorBoard:
```bash
python%load_ext tensorboard
%tensorboard --logdir=logs
```

### Model Outputs
The implementation saves various outputs:

- Model checkpoints in the `checkpoints/` directory
- Generated images in the `images/` directory
- TensorBoard logs in the `logs/` directory
- Comparative visualizations in the main `images/` folder

### Metrics Explanation

- Categorical Accuracy: Measures how well the model recovers discrete latent codes (higher is better)
- Continuous Correlation: Measures correlation between input continuous codes and their reconstructions (higher is better)
- Disentanglement Score: Quantifies how well each latent dimension exclusively controls one feature (higher is better)
- Factor Independence: Measures cross-talk between different latent factors (lower is better)
- Traversal Linearity: Measures the consistency of changes when traversing latent space (higher is better)

### Citation
```bibtex
If you use this code in your research, please cite:
bibtex@article{chen2016infogan,
  title={InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets},
  author={Chen, Xi and Duan, Yan and Houthooft, Rein and Schulman, John and Sutskever, Ilya and Abbeel, Pieter},
  journal={Advances in Neural Information Processing Systems},
  year={2016}
}
```
