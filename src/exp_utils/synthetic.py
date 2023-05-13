"""
# Created by ashish1610dhiman at 12/05/23
Contact at ashish1610dhiman@gmail.com
"""

import numpy as np
import tensorflow as tf
from src.edl import dense_loss,dense_layers
from src.weibull_edl import loss_and_layers
from scipy.special import loggamma


def gen_data_weibulll(x_min, x_max, n, eps=0.2, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(np.float32)
    # print (x.shape)
    if train:
        y = x**2 + eps*np.random.weibull(a=1.2,size=x.shape)
    else:
        y = x**2
    y = y.astype(np.float32)
    return x, y


def results_benchmark_model(c,x_train,y_train,x_test,verbose_fit=0):
    edl_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_dim=x_train.shape[1]),
        tf.keras.layers.Dense(200, activation="leaky_relu"),
        tf.keras.layers.Dense(200, activation="leaky_relu"),
        tf.keras.layers.Dense(200, activation="leaky_relu"),
        dense_layers.DenseNormalGamma(1),
    ])

    def EvidentialRegressionLoss(true, pred):
        return dense_loss.EvidentialRegression(true, pred, coeff=c)

    # Compile and fit the model!
    edl_model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=EvidentialRegressionLoss)

    edl_model.fit(x_train, y_train, batch_size=100, epochs=1000, verbose=verbose_fit)
    y_pred = edl_model(x_test).numpy()
    #get mu and variance
    mu, v, alpha, beta = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    var = np.sqrt(beta / (v * (alpha - 1)))
    return mu, var,  y_pred, edl_model


def results_weibull_model(c,x_train,y_train,x_test,k,verbose_fit=0):
    k=k
    weibull_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_dim=x_train.shape[1]),
        tf.keras.layers.Dense(200, activation="leaky_relu"),
        tf.keras.layers.Dense(200, activation="leaky_relu"),
        tf.keras.layers.Dense(200, activation="leaky_relu"),
        tf.keras.layers.Dense(200, activation="leaky_relu"),
        tf.keras.layers.Dense(200, activation="leaky_relu"),
        loss_and_layers.DenseWeibullGamma(1),
    ])

    def weibullLoss(true, pred):
        return loss_and_layers.weibull_evidence_Regression(true, pred, k=k, coeff=c)

    # Compile and fit the model!
    weibull_model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=weibullLoss)

    weibull_model.fit(x_train, y_train, batch_size=100, epochs=1000, verbose = verbose_fit)
    y_pred = weibull_model.predict(x_test);
    alpha,beta = y_pred[:,0],y_pred[:,1]
    mean_pred_log = (loggamma(1 + (1 / k)) - loggamma(alpha) + loggamma(alpha - (1 / k)) \
                     + (1 / k) * np.log(beta))
    mu = np.exp(mean_pred_log)
    var_term1 = (2*loggamma(1 + (1/ k)) - loggamma(alpha) + loggamma(alpha - (2 / k)) \
                 + (2 / k) * np.log(beta))
    var = np.exp(var_term1) - np.square(mu)
    return mu, var, y_pred, weibull_model
