# coding:utf-8

import tensorflow as tf
from tensorflow.python.layers import core as layers_core


# 如文件名所述， 就是一个 MLP 层。

class MLP:
    """
    Easy-to-use mlp class
    """

    def __init__(self, num_hidden_layers,
                 dropout=True,
                 dropout_rate=0.2,
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 name=None,
                 reuse=False):

        self.layers = []
        self.num_hidden_layers = num_hidden_layers
        self._name = name if name else 'MLP'
        self._reuse = reuse

        for zb_layer_num, num_hidden_layer in enumerate(num_hidden_layers):
            layer_num = zb_layer_num + 1

            if dropout and layer_num < len(num_hidden_layers):
                self.layers.append(
                    layers_core.Dropout(rate=dropout_rate) ) # The dropout rate
                layer_activation = activation
            else:
                layer_activation = None

            self.layers.append(
                layers_core.Dense(
                    num_hidden_layer,  # Dense dim ;   output_dim, 前一维度，自动推断；
                    activation=layer_activation, 
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    trainable=True,
                    name='{}/dense_{}'.format(self._name, layer_num),
                    _reuse=reuse
                )
            )
    
    def apply(self, inputs, is_training=True):
        for layer in self.layers:
            if isinstance(layer, layers_core.Dropout):   # isinstance 判断两个对象 是否是同一个类型
                inputs = layer.apply(inputs, training=is_training)
            else:
                inputs = layer.apply(inputs)
        final_output = inputs
        return final_output

