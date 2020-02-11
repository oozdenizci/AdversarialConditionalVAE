# Adversarially Censored Conditional Variational Autoencoders

This is an implementation of adversarially censored conditional variational autoencoder (cVAE) models for invariant latent space learning. Conventional cVAE model learning is performed within an adversarial training setting to monitor and/or censor nuisance-specific leakage in the learned latent space. Implementation is in Python using Keras with Tensorflow backend, and was intended to be used for electroencephalographic (EEG) data. This is a re-implementation of presented work with arbitrary sub-network architectures: https://dx.doi.org/10.1109/NER.2019.8716897

# Usage

An example execution is as follows:

```python

from AdversarialcVAE import AdversarialcVAE

net = AdversarialcVAE(chans = ..., samples = ..., n_latent = ..., n_nuisance = ..., n_kernels = ..., adversarial = ..., lam = ...)

net.train(train_set, validation_set, log = ..., epochs = ..., batch_size = ...)

```

Parameter `n_latent` defines the latent space dimensionality, `n_nuisance` defines the nuisance variable dimensionality (i.e., the condition and the adversary output), and `n_kernels` define the number of convolutional kernel filters within the encoder and decoder which both have arbitrary structures for now. Boolean parameter `adversarial = True` trains the cVAE via adversarial censoring. If `False`, then an adjacent adversary network is simply trained to monitor nuisance-specific leakage in the latent representations. Parameter `lam` indicates the adversarial regularization weight embedded in the total loss function.

To use the `train` function, both `train_set` and `validation_set` should be three-element tuples (i.e., `x_train, y_train, s_train = train_set`). Here, the first element `x_train` is the EEG data of size `(num_observations, num_channels, num_timesamples, 1)`, `y_train` are the one-hot encoded class labels (e.g., for binary labels will have a size `(num_observations, 2)`), and `s_train` are the one-hot encoded nuisance labels used for censoring (e.g., for 10-class nuisance labels will have size `(num_observations, 10)`). Variable `log` indicates the directory string to save the log files during training.

# Paper Citation
If you use this code in your research and find it helpful, please cite the following paper:
> Ozan Ozdenizci, Ye Wang, Toshiaki Koike-Akino, Deniz Erdogmus. "Transfer learning in brain-computer interfaces with adversarial variational autoencoders". 9th International IEEE/EMBS Conference on Neural Engineering (NER), 2019. https://dx.doi.org/10.1109/NER.2019.8716897
