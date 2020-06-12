Github Repository for Learning Portrait Style Representations. Submitted for Review to Vision for Art Visart V Workshop.

# Setting up
1. To download necessary packages run:
```pip3 install -r requirements.txt```

2. Download the Image data from <INSERT WEBSITE URL HERE> and put it into your project root directory

3. When running either `triplet_vgg.py`, `vae_no_trip.py`, or `vae_trip.py`, first open the respective file, and modify `mount_filepath` to the path to your project root directory. Change the variable "model_name_prefix" to your experiment name.

4. To run, say `triplet_vgg.py`, run `python3 triplet_vgg.py triplet_vgg.yaml`. For configuration options, see `triplet_vgg.yaml` and modify to your needs.

# Experiments

1. `triplet_vgg.py` runs the VGG-16 + Triplet loss experiment described in the paper.
2. `vae_no_trip.py` runs the VAE experiment described in the paper.
3. `vae_trip.py`    runs the VAE + Triplet loss experiment described in the paper.

# Hyperparameters

Train-Test Split for all experiments was 70-30 train/test.

VGG-16 + Triplet Loss
| Hyperparameter | Value | Notes |
| :---: | :---: | :---: |
| # of Style Encoding Layers | 2 | 4096 VGG Feature Embedding -> 2048 -> 1024 Style Embedding |
| Triplet Loss Ratio | 1.0 | |

VAE
| Hyperparameter | Value | Notes |
| :---: | :---: | :---: |
| # of Encoding Layers | 8 | 8 Stride-2 Convolution Layers|
| Latent Vector Size | 1024 | Size of the Latent Style Vector |
| # of Decoding Layers | 8 | 8 Transpose Convolution Layers|
| Perceptual Loss Ratio | .1 | Ratio of Percep Loss to other losses|
| KLD Loss Ratio | .001 | Ratio of KLD Loss to other losses|
| Reconstruction Loss Ratio | 1 | Ratio of Reconstruction Loss to other losses|

VAE + Triplet Loss

| Hyperparameter | Value | Notes |
| :---: | :---: | :---: |
| # of Encoding Layers | 8 | 8 Stride-2 Convolution Layers|
| Latent Vector Size | 1024 | Size of the Latent Style Vector |
| # of Decoding Layers | 8 | 8 Transpose Convolution Layers|
| Perceptual Loss Ratio | .1 | Ratio of Percep Loss to other losses|
| Triplet Loss Ratio | .1 | Ratio of Triplet Loss to other losses|
| KLD Loss Ratio | .001 | Ratio of KLD Loss to other losses|
| Reconstruction Loss Ratio | 1 | Ratio of Reconstruction Loss to other losses|




