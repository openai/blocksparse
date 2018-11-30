"""Block sparse layer.
"""

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops import nn
from blocksparse.matmul import BlocksparseMatMul

class BlockSparse(Layer):
    """ A blocksparse variant of a regular densely-connected NN layer.
    
    `BlockSparse` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, 'dot' is a blocksparse variant
    of the dot product as defined in `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    
    Example:
    
    ```python
        model = Sequential()
        model.add(Dense(32, BarabasiAlbert(2), input_shape=(16,)))
      # now the model will take as input arrays of shape (*, 16)
      # and output arrays of shape (*, 32)
      # after the first layer, you don't need to specify
      # the size of the input anymore:
      model.add(Dense(32), BarabasiAlbert(2))
    ```
    
    Arguments:
        units: Positive integer, dimensionality of the output space.
        sparsity_mask_initializer: Initializer for the sparsity mask
          of the `kernel` weights matrix.
        blocksize: values 32, 16, 8 supported
        feature_axis Boolean, when block_size is less than 32 memory access becomes far more efficient
          with a (C,N) activation layout
        activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        
    Input shape:
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        a 2D input with shape `(batch_size, units)`.
    """
    
    def __init__(self,
                 units,
                 sparsity_mask_initializer,
                 blocksize=32,
                 feature_axis=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        
        super(BlockSparse, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.units = units
        self.sparsity_mask_initializer = sparsity_mask_initializer
        self.blocksize = blocksize
        self.feature_axis = feature_axis
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.sparsity_mask = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
            
        if self.units%self.blocksize != 0:
            raise ValueError('The number of units should be divisible by the blocksize. '
                             'Got {} units and blocksize {}'.format(self.units, self.blocksize))
        if input_shape[-1].value%self.blocksize != 0:
            raise ValueError('The input_shape should be divisible by the blocksize. '
                             'Got {} units and blocksize {}'.format(input_shape[-1].value, self.blocksize))
        
        mask_shape=(input_shape[-1].value//self.blocksize, self.units//self.blocksize)
        
        self.sparsity_mask = self.sparsity_mask_initializer(mask_shape)

        self.bsmm = BlocksparseMatMul(self.sparsity_mask,
                                      block_size=self.blocksize,
                                      feature_axis=self.feature_axis,
                                      name=self.name + '_bsmm')
        
        self.kernel = self.add_weight('kernel',
                                      shape=self.bsmm.w_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
        
        if self.use_bias:
            self.bias = self.add_weight('bias',
                            shape=[self.units,],
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                            dtype=self.dtype,
                            trainable=True)

    def call(self, input):
        if self.feature_axis:
            outputs = self.bsmm(input, self.kernel)
        else:
            outputs = self.bsmm(tf.transpose(input), self.kernel)
            outputs = tf.transpose(outputs)
        
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)