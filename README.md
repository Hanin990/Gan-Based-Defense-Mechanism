# InfoGAN-based Defense Mehcanism

A defense mechanism against adversarial attacks on NLP text classification models using InfoGAN-based embedding reconstruction.

## Overview

This repository implements the **InfoGAN-based Defense Mehcanism** approach for defending transformer-based text classifiers against adversarial examples. The defense leverages InfoGAN generative models to reconstruct perturbed embeddings back onto the learned textual manifold, effectively neutralizing adversarial perturbations.

Based on the research: *"Textual Manifold-based Defense Against Natural Language Adversarial Examples"*

## Key Features

- **InfoGAN-based Embedding Reconstruction**: Uses Information-Maximizing GAN to learn the manifold of clean text embeddings
- **Multiple Reconstruction Strategies**: 8 different methods for finding optimal reconstructions:
  - Method 1: SGD optimization in latent space
  - Method 3: Random sampling from normal prior
  - Method 4: Truncated normal sampling (default)
  - Method 5: Single truncated sample
  - Method 6-7: Constrained optimization with encoder guidance
  - Method 8: Joint z and c prior sampling
- **Support for Multiple Language Models**:
  - BERT / BERT-Large
  - RoBERTa / RoBERTa-Large
  - XLNet
  - ELECTRA
  - ALBERT
- **Integrated Adversarial Attack Benchmarking**: Built-in support for evaluating against multiple attack methods
- **Weights & Biases Integration**: Comprehensive experiment tracking and visualization

## Project Structure

```
Gan-Based-Defense-Mechanism/
├── main.py                          # Main entry point for attack/evaluate modes
├── args.py                          # Argument parsing and configuration
├── src/
│   ├── train/
│   │   └── train.py                 # InfoGAN training script
│   ├── models/
│   │   ├── __init__.py
│   │   ├── language_models/         # Transformer model wrappers
│   │   │   ├── lm.py                # Base language model interface
│   │   │   ├── auto_lm.py           # Auto model selection
│   │   │   └── huggingface/         # HuggingFace implementations
│   │   │       ├── bert.py
│   │   │       ├── roberta.py
│   │   │       └── xlnet.py
│   │   └── generative_models/       # Generative defense models
│   │       ├── gm.py                # Base generative model
│   │       ├── auto_gm.py           # Auto model selection
│   │       ├── gan_based/
│   │       │   ├── gan.py           # Base GAN
│   │       │   └── infogan.py       # InfoGAN implementation
│   │       └── vae_based/
│   │           ├── vae.py
│   │           └── vae_gan.py
│   └── data/
│       ├── datamodule.py            # Data loading utilities
│       └── encoded_datamodule.py    # Pre-encoded data handling
├── textattack/                      # TextAttack framework integration
├── trainer/                         # Training strategies
├── utils/
│   ├── config.py                    # Model and dataset configurations
│   ├── textattack_utils.py          # Attack builder utilities
│   └── metrics.py                   # Evaluation metrics
├── data/                            # Dataset storage
├── install_env.sh                   # Environment setup script
└── setup.sh                         # Dependencies installation
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Miniconda/Anaconda (optional but recommended)

### Setup Steps

1. **Create a conda environment** (optional):
   ```bash
   bash install_env.sh
   ```
   This will install Miniconda and create a Python 3.8 environment named `myenv`.

2. **Install dependencies**:
   ```bash
   bash setup.sh
   ```

   Or install manually:
   ```bash
   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/
   pip install transformers==4.25.1
   pip install pytorch-lightning
   pip install wandb
   pip install datasets
   pip install evaluate
   pip install textattack
   pip install pandas==1.2.5
   pip install numpy==1.19.5
   pip install matplotlib==3.5
   pip install scipy
   pip install overrides
   pip install colorama tabulate psutil scikit-learn
   ```

3. **Login to Weights & Biases** (for experiment tracking):
   ```bash
   wandb login
   ```

## Usage

### Training InfoGAN Defense Model

Train the InfoGAN generative model on clean embeddings from your fine-tuned language model:

```bash
python src/train/train.py \
   --data=hard \
   --data_root=/path/to/data/hard \
   --lm=bert \
   --lm_path=/path/to/finetuned-bert-model \
   --gm=infogan \
   --num_epochs=200 \
   --g_lr=5e-5 \
   --d_lr=2e-4 \
   --reg_prior_weight=5.0 \
   --reg_info_weight=0.5 \
   --k=10 \
   --feature_matching_weight=0.05 \
   --grad_clip_norm=0.5 \
   --accelerator=gpu \
   --devices=1 \
   --output_dir=/path/to/output
```

**Key Training Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Dataset name | `hard` |
| `--lm` | Language model type (`bert`, `roberta`, `xlnet`) | `bert` |
| `--lm_path` | Path to fine-tuned language model | - |
| `--gm` | Generative model type | `infogan` |
| `--num_epochs` | Number of training epochs | `200` |
| `--g_lr` | Generator learning rate | `1e-4` |
| `--d_lr` | Discriminator learning rate | `4e-4` |
| `--reg_prior_weight` | Prior regularization weight | `5.0` |
| `--reg_info_weight` | Information loss weight | `0.5` |
| `--k` | Number of reconstruction candidates | `15` |
| `--grad_clip_norm` | Gradient clipping norm | `1.0` |

### Running Adversarial Attacks

Evaluate model robustness with TMD defense against various attacks:

```bash
python main.py \
   --mode=attack \
   --model_type=bert \
   --model_name_or_path=/path/to/finetuned-bert-model \
   --dataset_name=hard \
   --dataset_path=/path/to/dataset/hard \
   --training_type=tmd \
   --gm=infogan \
   --gm_path=/path/to/trained-infogan.ckpt \
   --tmd_layer=-1 \
   --max_seq_len=128 \
   --do_lower_case=True \
   --attack_method=pwws \
   --method=4 \
   --k=20 \
   --threshold=1.0 \
   --neighbour_vocab_size=10 \
   --modify_ratio=0.2 \
   --start_index=0 \
   --end_index=2000
```

**Key Attack Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Run mode (`attack`, `evaluate`) | `attack` |
| `--model_type` | Language model type | `bert` |
| `--training_type` | Defense type (`tmd`, `none`) | `none` |
| `--gm` | Generative model for defense | `infogan` |
| `--gm_path` | Path to trained generative model | - |
| `--tmd_layer` | Layer for TMD (-1 for last) | `-1` |
| `--attack_method` | Attack algorithm | `pwws` |
| `--method` | Reconstruction method (1-8) | `4` |
| `--k` | Number of reconstruction samples | `20` |
| `--threshold` | Truncation threshold for method 4/5 | `1.0` |
| `--modify_ratio` | Max word modification ratio | `0.2` |

### Evaluating Model Performance

Run evaluation without attacks:

```bash
python main.py \
   --mode=evaluate \
   --model_type=bert \
   --model_name_or_path=/path/to/bert-model \
   --dataset_name=hard \
   --training_type=tmd \
   --gm=infogan \
   --gm_path=/path/to/infogan.ckpt
```

## Supported Datasets

| Dataset | Labels | Task |
|---------|--------|------|
| AG News | 4 | News Classification |
| SST-2 | 2 | Sentiment Analysis |
| IMDB | 2 | Sentiment Analysis |
| Yelp Polarity | 2 | Sentiment Analysis |
| SNLI | 3 | Natural Language Inference |
| MR | 2 | Movie Reviews |
| Custom (hard/bard) | 2 | Binary Classification |

## Supported Attacks

| Attack | Method | Reference |
|--------|--------|-----------|
| **PWWS** | Word importance + WordNet synonyms | Ren et al., 2019 |
| **TextFooler** | Word importance + embedding neighbors | Jin et al., 2019 |
| **BERT-Attack (BAE)** | BERT masked LM substitutions | Li et al., 2020 |
| **PSO** | Particle Swarm Optimization | Zang et al., 2020 |
| **Genetic Algorithm** | Evolutionary word substitution | Alzantot et al., 2018 |
| **DeepWordBug** | Character-level perturbations | Gao et al., 2018 |
| **TextBugger** | Character + word-level attacks | Li et al., 2018 |
| **HotFlip** | Gradient-based character flips | Ebrahimi et al., 2017 |

## Model Architecture

### TMD Defense Pipeline

```
Input Text → Tokenizer → Language Model → Embedding (Layer -1)
                                              ↓
                              [If TMD enabled] InfoGAN Reconstructor
                                              ↓
                              Reconstructed Embedding → Classifier → Prediction
```

### InfoGAN Components

1. **Generator**: Maps latent code (z, c) to embedding space
   - Progressive architecture with residual connections
   - LayerNorm + ReLU activations
   - Dropout for regularization

2. **Discriminator**: Distinguishes real/fake embeddings + encodes latent code
   - Spectral normalization for training stability
   - Separate heads for adversarial and information losses

3. **Prior Network**: Learnable categorical prior distribution

### Reconstruction Process

The defense reconstructs potentially adversarial embeddings:

1. Encode input to get discrete code c via discriminator
2. Sample k candidates from latent space z
3. Generate fake embeddings G(z, c) for each candidate
4. Select reconstruction closest to original embedding
5. Use reconstructed embedding for classification

## Citation

If you use this code, please cite:

```bibtex
@article{tmd-defense,
  title={Textual Manifold-based Defense Against Natural Language Adversarial Examples},
  author={...},
  journal={...},
  year={...}
}
```

## License

This project is provided for research purposes.

## Acknowledgments

- [TextAttack](https://github.com/QData/TextAttack) framework for adversarial attack implementations
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for pre-trained language models
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning) for training infrastructure
