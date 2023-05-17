"""
# Created by ashish1610dhiman at 16/05/23
Contact at ashish1610dhiman@gmail.com
"""

"""
# Created by ashish1610dhiman at 12/05/23
Contact at ashish1610dhiman@gmail.com
"""

import numpy as np
import tensorflow as tf
from src.edl import dense_loss,dense_layers
from src.weibull_edl import loss_and_layers
from scipy.special import loggamma,gamma
from sklearn.metrics import mean_squared_error


def metrics_benchmark(y_true,y_pred):
    gamma, v, alpha, beta = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    mse = mean_squared_error(y_true, gamma)
    nll = dense_loss.NIG_NLL(y_true,gamma,v,alpha,beta).numpy()
    return (mse,nll)

def metrics_proposed(y_true,y_pred,k):
    alpha,beta = y_pred[:,0],y_pred[:,1]
    mean_pred_log = (loggamma(1 + (1 / k)) - loggamma(alpha) + loggamma(alpha - (1 / k)) \
                     + (1 / k) * np.log(beta))
    mu = np.exp(mean_pred_log)
    mse = mean_squared_error(y_true, mu)
    nll = loss_and_layers.weibull_NLL(y_true,alpha,beta, k, reduce=True).numpy()
    return (mse,nll)

def results_benchmark_model(c,x_train,y_train,x_test,verbose_fit=0):
    edl_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_dim=x_train.shape[1]),
        tf.keras.layers.Dense(350, kernel_initializer='normal', activation="leaky_relu"),
        tf.keras.layers.Dense(300, kernel_initializer='normal', activation="relu"),
        tf.keras.layers.Dense(300, kernel_initializer='normal', activation="relu"),
        tf.keras.layers.Dense(250, kernel_initializer='normal', activation="leaky_relu"),
        tf.keras.layers.Dense(250, kernel_initializer='normal', activation="relu"),
        tf.keras.layers.Dense(200, kernel_initializer='normal', activation="relu"),
        tf.keras.layers.Dense(200, kernel_initializer='normal', activation="leaky_relu"),
        tf.keras.layers.Dense(200, kernel_initializer='normal', activation="leaky_relu"),
        dense_layers.DenseNormalGamma(1),
    ])

    def EvidentialRegressionLoss(true, pred):
        return dense_loss.EvidentialRegression(true, pred, coeff=c)

    # Compile and fit the model!
    edl_model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-5),
        loss=EvidentialRegressionLoss)

    history = edl_model.fit(x_train, y_train, batch_size=120, epochs=340, verbose=verbose_fit)
    y_pred_train = edl_model.predict(x_train)
    y_pred_test = edl_model.predict(x_test)
    #get mu and variance
    mu, v, alpha, beta = y_pred_test[:, 0], y_pred_test[:, 1], y_pred_test[:, 2], y_pred_test[:, 3]
    var = np.sqrt(beta / (v * (alpha - 1)))
    return mu, var,  y_pred_train, y_pred_test, edl_model, history


def results_weibull_model(c,x_train,y_train,x_test,k,verbose_fit=0):
    k=k
    weibull_model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_dim=x_train.shape[1]),
        tf.keras.layers.Dense(350, activation="leaky_relu"),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(250, activation="leaky_relu"),
        tf.keras.layers.Dense(250, activation="relu"),
        tf.keras.layers.Dense(200, activation="relu"),
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

    history = weibull_model.fit(x_train, y_train, batch_size=88, epochs=340, verbose = verbose_fit)
    y_pred_train = weibull_model.predict(x_train)
    y_pred_test = weibull_model.predict(x_test)
    alpha,beta = y_pred_test[:,0],y_pred_test[:,1]
    mean_pred_log = (loggamma(1 + (1 / k)) - loggamma(alpha) + loggamma(alpha - (1 / k)) \
                     + (1 / k) * np.log(beta))
    mu = np.exp(mean_pred_log)
    # var_term1 = (2 * loggamma(1 + (1 / k))) + (loggamma(alpha - (2 / k))) + ((2 / k) * np.log(beta)) - loggamma(alpha)
    # var = np.exp(var_term1)-np.square(mu)
    # var[var<0.00] = var[var<0.00].min() #numerical consistency
    var_term1 = (loggamma(alpha - (2 / k))) + ((2 / k) * np.log(beta)) - loggamma(alpha)
    var_term2 = loggamma(1 + (2/ k)) -np.square(loggamma(1 + (1/ k)))
    var = np.exp(var_term1 + var_term2)
    # var_term1 = (2*loggamma(1 + (2/ k))) + ((2 / k) * np.log(beta)) - loggamma(alpha)
    # var_term2 = gamma(alpha - (2 / k)) - (np.square(gamma(alpha - (1 / k))))/gamma(alpha)
    # var_term1 = loggamma(1 + (2/ k)) - np.square(loggamma(1 + (1/ k)))
    # var = np.exp(var_term1 - loggamma(alpha) + loggamma(alpha - (2 / k)) + (2 / k) * np.log(beta))
    return mu, var, y_pred_train ,y_pred_test, weibull_model, history