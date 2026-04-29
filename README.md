# Generative Video Forecasting

This repository implements several approaches for learned forecasting of small video datasets. The shared Python package under `src/video_forecasting/` contains dataset loaders/generators, VAE variants, forecasting models, training loops, runtime helpers, and visualization code.


The current datasets are:

- **Moving MNIST**, introduced in [this paper](https://doi.org/10.48550/arXiv.1502.04681).
- **Elastic disk dynamics**, a generated dataset of grayscale movies showing circular particles moving in a 2D box with reflecting or periodic boundaries. The simulation uses simple equal-mass elastic collisions and wall/periodic boundary handling. `ElasticDisksDataset` supports configurable `image_size`; Moving MNIST uses its native `64x64` resolution.

| Moving MNIST | Elastic disk dynamics |
|:------------:|:---------------------:|
| ![Sample Moving MNIST sequence](moving_mnist_sequence.gif) | ![Sample elastic disk sequence](elastic_disks_sequence.gif) |

<p align="center">
  <img src="elastic_disks_sequence_0_simvp_rollout.gif" width="384" alt="Autoregressive SimVP rollout: prediction, ground truth, and error (Elastic Disks, sequence 0)." />
</p>
<p align="center"><em>Autoregressive SimVP rollout conditioned on observed context frames for a sample from the elastic disks dataset. Left: model-generated frames. Center: ground truth frames. Right: error map.</em></p>

## Notebooks

```text
notebooks/
  moving_mnist/
    visualize_moving_mnist_data.ipynb
    train_moving_mnist_pixel_flow_matching.ipynb
    train_moving_mnist_pixel_stochastic_interpolant.ipynb
    train_moving_mnist_pixel_diffusion.ipynb
    train_moving_mnist_simvp.ipynb
    train_moving_mnist_latent_flow_matching.ipynb
    train_moving_mnist_latent_stochastic_interpolant.ipynb
    train_moving_mnist_latent_flow_matching_1dlatent.ipynb
    train_moving_mnist_latent_diffusion_1dlatent.ipynb
    train_moving_mnist_latent_transformer.ipynb
    train_moving_mnist_mdn_rnn.ipynb
    train_moving_mnist_mdn_rnn_1dvaelatent.ipynb
  elastic_disks/
    visualize_elastic_disks_data.ipynb
    train_elastic_disks_pixel_flow_matching.ipynb
    train_elastic_disks_pixel_stochastic_interpolant.ipynb
    train_elastic_disks_pixel_diffusion.ipynb
    train_elastic_disks_simvp.ipynb
    train_elastic_disks_latent_flow_matching.ipynb
    train_elastic_disks_latent_stochastic_interpolant.ipynb
    train_elastic_disks_latent_flow_matching_1dlatent.ipynb
    train_elastic_disks_latent_diffusion_1dlatent.ipynb
    train_elastic_disks_latent_transformer.ipynb
    train_elastic_disks_mdn_rnn.ipynb
    train_elastic_disks_mdn_rnn_1dvaelatent.ipynb
```

The forecasting notebooks cover pixel- and latent-space flow matching, pixel- and latent-space stochastic interpolants, pixel- and latent-space diffusion, SimVP, causal latent transformers, MDN-RNNs over spatial latents, and MDN-RNNs over 1D VAE latents. 


## Package Layout

```text
src/video_forecasting/
  datasets/
    moving_mnist.py
    elastic_disks.py
  models/
    vae.py
    flow_matching.py
    stochastic_interpolants.py
    diffusion.py
    transformer.py
    mdn_rnn.py
    simvp.py
  runtime.py
  training.py
  visualization.py
```
