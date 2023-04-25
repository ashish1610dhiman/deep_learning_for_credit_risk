"""
# Created by ashish1610dhiman at 22/04/23
Contact at ashish1610dhiman@gmail.com
"""

import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense
from tensorflow.math import log as log_tf

class DenseWeibullGamma(Layer):
    def __init__(self, units):
        super(DenseWeibullGamma, self).__init__()
        self.units = int(units)
        self.dense = Dense(2 * self.units, activation=None)

    def evidence(self, x):
        # return tf.exp(x)
        return tf.nn.softplus(x)

    def call(self, x):
        output = self.dense(x)
        logalpha, logbeta = tf.split(output, 2, axis=-1)
        alpha = self.evidence(logalpha) + 1.0
        beta = self.evidence(logbeta)
        return tf.concat([alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.units)

    def get_config(self):
        base_config = super(DenseWeibullGamma, self).get_config()
        base_config['units'] = self.units
        return base_config

def weibull_NLL(y, alpha, beta,k, reduce=False):
    nll = -(log_tf(alpha) + log_tf(k) + (k-1)* log_tf(y)\
            + (alpha * log_tf(beta)) - (alpha+1)*log_tf(y + beta))
    return tf.reduce_mean(nll) if reduce else nll

def ad_Reg(y,alpha, beta,k, reduce=True):
    # pred_mean_log = tf.math.lgamma((k*alpha-1)/k) - log_tf(k) - ((alpha-1)/k)*log_tf(beta)
    pred_mean_log = (tf.math.lgamma(1+ (1/k)) - tf.math.lgamma(alpha) + tf.math.lgamma(alpha-(1/k))\
                + (1/k)*log_tf(beta))
    # error = tf.stop_gradient(tf.abs(y-gamma))
    error = tf.abs(y-tf.math.exp(pred_mean_log))
    evi = (alpha)
    reg = error*evi
    return reg


def weibull_evidence_Regression(y_true, evidential_output, k, coeff=1.0):
    alpha, beta = tf.split(evidential_output, 2, axis=-1)
    # print (gamma, v ,alpha, beta)
    loss_nll = weibull_NLL(y_true, alpha, beta, k)
    loss_reg = ad_Reg(y_true, alpha, beta, k)
    return loss_nll + coeff*loss_reg
