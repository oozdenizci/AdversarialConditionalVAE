#!/usr/bin/env python
import keras
import keras.backend as K
from keras.layers import Input, Dense, Lambda, Flatten, BatchNormalization
from keras.layers import Activation, Reshape, AveragePooling2D, UpSampling2D
from keras.layers.convolutional import Conv2D, Deconvolution2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error
import numpy as np
keras.backend.set_image_data_format('channels_last')


class AdversarialcVAE:
    def __init__(self, chans, samples, n_latent, n_nuisance, n_kernels, adversarial=False, lam=0.01):

        # Input, data set and model training scheme parameters
        self.chans = chans
        self.samples = samples
        self.n_latent = n_latent
        self.n_nuisance = n_nuisance
        self.n_kernels = n_kernels
        self.lam = lam

        # Build the network blocks
        self.enc = self.encoder_model()
        self.dec = self.decoder_model()
        self.adv = self.adversary_model()

        # Compile the network with or without adversarial censoring
        x = Input(shape=(self.chans, self.samples, 1))
        s = Input(shape=(self.n_nuisance, ))
        z_mu, z_log_var, z = self.enc(x)
        x_hat = self.dec([z, s])
        s_hat = self.adv(z)
        self.adv.trainable = False
        self.acvae = Model([x, s], [x_hat, s_hat])

        def vae_loss(y_true, y_pred):
            mse = mean_squared_error(y_true, y_pred)
            reconstruction_loss = K.mean(K.mean(mse, axis=1), axis=1)
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mu) - K.exp(z_log_var), axis=-1)
            return K.mean(reconstruction_loss + kl_loss)

        if adversarial:
            self.acvae.compile(loss=[vae_loss, 'categorical_crossentropy'],
                               loss_weights=[1., -1. * self.lam], optimizer=Adam(lr=1e-3, decay=1e-4))
        else:
            self.acvae.compile(loss=[vae_loss, 'categorical_crossentropy'],
                               loss_weights=[1., 0.], optimizer=Adam(lr=1e-3, decay=1e-4))

        self.adv.trainable = True
        self.adv.compile(loss=['categorical_crossentropy'], optimizer=Adam(lr=1e-3, decay=1e-4), metrics=['accuracy'])

    def encoder_model(self):
        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=(K.shape(mu)[0], K.int_shape(mu)[1]))
            return mu + K.exp(0.5 * log_var) * epsilon

        x_in = Input(shape=(self.chans, self.samples, 1))
        x = Conv2D(self.n_kernels, (1, 40), padding='same', use_bias=False)(x_in)
        x = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(x)
        x = Activation('elu')(x)
        x = Conv2D(self.n_kernels, (self.chans, 1), use_bias=False)(x)
        x = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(x)
        x = Activation('elu')(x)
        x = AveragePooling2D(pool_size=(1, 2))(x)
        x = Flatten()(x)
        z_mu = Dense(self.n_latent, use_bias=False)(x)
        z_log_var = Dense(self.n_latent, use_bias=False)(x)
        z = Lambda(sampling, name='latent')([z_mu, z_log_var])

        return Model(x_in, [z_mu, z_log_var, z], name='enc')

    def decoder_model(self):
        z_in = Input(shape=(self.n_latent,))
        s_in = Input(shape=(self.n_nuisance,))
        z = Dense(int(self.samples//2 * self.n_kernels), use_bias=False)(concatenate([z_in, s_in]))
        z = Reshape((1, int(self.samples//2), self.n_kernels))(z)
        z = UpSampling2D(size=(1, 2))(z)
        z = Deconvolution2D(self.n_kernels, (self.chans, 1), use_bias=False)(z)
        z = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(z)
        z = Activation('elu')(z)
        z = Deconvolution2D(1, (1, 40), padding='same', use_bias=False)(z)
        z = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(z)
        x_hat = Activation('elu')(z)

        return Model([z_in, s_in], x_hat, name='dec')

    def adversary_model(self):
        z_in = Input(shape=(self.n_latent,))
        z = Dense(self.n_nuisance, use_bias=False)(z_in)
        s_hat = Activation('softmax')(z)

        return Model(z_in, s_hat, name='adv')

    def train(self, train_set, val_set, log, epochs=500, batch_size=50):
        x_train, y_train, s_train = train_set
        x_test, y_test, s_test = val_set

        train_index = np.arange(x_train.shape[0])
        train_batches = [(i * batch_size, min(x_train.shape[0], (i + 1) * batch_size))
                         for i in range((x_train.shape[0] + batch_size - 1) // batch_size)]

        # Early stopping variables
        es_wait = 0
        es_best = np.Inf
        es_best_weights = None

        for epoch in range(1, epochs + 1):
            print('Epoch {}/{}'.format(epoch, epochs))
            np.random.shuffle(train_index)
            train_log, train_log_adv = [], []
            for iter, (batch_start, batch_end) in enumerate(train_batches):
                batch_ids = train_index[batch_start:batch_end]
                x_train_batch = x_train[batch_ids]
                s_train_batch = s_train[batch_ids]
                z_mu_train_batch, z_log_var_train_batch, z_train_batch = self.enc.predict_on_batch(x_train_batch)

                train_log_adv.append(self.adv.train_on_batch(z_train_batch, s_train_batch))
                train_log.append(self.acvae.train_on_batch([x_train_batch, s_train_batch], [x_train_batch, s_train_batch]))
            train_log = np.mean(train_log, axis=0)
            train_log_adv = np.mean(train_log_adv, axis=0)

            val_log = self.acvae.test_on_batch([x_test, s_test], [x_test, s_test])
            z_mu_test, z_log_var_test, z_test = self.enc.predict_on_batch(x_test)
            val_log_adv = self.adv.test_on_batch(z_test, s_test)

            # Logging model training information per epoch
            print("Train - [Loss: %f] - [ADV loss: %f, acc: %.2f%%]" % (train_log[0], train_log_adv[0], 100*train_log_adv[1]))
            print("Validation - [Loss: %f] - [ADV loss: %f, acc: %.2f%%]" % (val_log[0], val_log_adv[0], 100*val_log_adv[1]))
            with open(log + '/train.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(train_log[0]) + ',' + str(train_log_adv[0]) + ',' + str(100*train_log_adv[1]) + '\n')
            with open(log + '/validation.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(val_log[0]) + ',' + str(val_log_adv[0]) + ',' + str(100*val_log_adv[1]) + '\n')

            # Check early stopping criteria based on validation loss - patience for 10 epochs
            if np.less(val_log[0], es_best):
                es_wait = 0
                es_best = val_log[0]
                es_best_weights = self.acvae.get_weights()
            else:
                es_wait += 1
                if es_wait >= 10:
                    print('Early stopping...')
                    self.acvae.set_weights(es_best_weights)
                    return
