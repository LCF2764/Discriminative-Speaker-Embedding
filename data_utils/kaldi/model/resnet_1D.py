import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from model.pooling import self_attention
from model.common import prelu, shape_list
from collections import OrderedDict
from six.moves import range


def conv_block(input_features, kernel_sizes, filters, strides, params, is_training, relu, name):
    """ ResNet Conv Block with shortcut. The shortcut is projected by an extra conv+bn.

    Args:
        input_features: input features
        kernel_sizes: list. length is 2 or 3
        filters: list.
        strides: The stride applied to the first conv.
        params:
        is_training:
        relu: inherited from the network to define the relu function
        name:
    :return:
    """
    n_comp = len(kernel_sizes)
    assert(n_comp == len(filters) and (n_comp == 2 or n_comp == 3))
    feat = tf.layers.conv1d(input_features,
                            filters[0],
                            kernel_sizes[0],
                            strides=strides,
                            padding='same',
                            activation=None,
                            use_bias=False,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                            name=name+"_conv0")
    feat = tf.layers.batch_normalization(feat,
                                         momentum=params.batchnorm_momentum,
                                         training=is_training,
                                         name=name+"_bn0")
    feat = relu(feat, name=name+'_relu0')

    feat = tf.layers.conv1d(feat,
                            filters[1],
                            kernel_sizes[1],
                            padding='same',
                            activation=None,
                            use_bias=False,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                            name=name+"_conv1")
    feat = tf.layers.batch_normalization(feat,
                                         momentum=params.batchnorm_momentum,
                                         training=is_training,
                                         name=name + "_bn1")

    if n_comp == 3:
        feat = relu(feat, name=name + '_relu1')
        feat = tf.layers.conv1d(feat,
                                filters[2],
                                kernel_sizes[2],
                                padding='same',
                                activation=None,
                                use_bias=False,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                name=name+"_conv2")
        feat = tf.layers.batch_normalization(feat,
                                             momentum=params.batchnorm_momentum,
                                             training=is_training,
                                             name=name + "_bn2")

    # Shortcut
    # When performing the shortcut projection, the kernel size is 1.
    # The #kernels equal to the last conv and the stride is consistent with the block's stride.
    shortcut = tf.layers.conv1d(input_features,
                                filters[-1],
                                1,
                                strides=strides,
                                padding='same',
                                activation=None,
                                use_bias=False,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                name=name + "_conv_short")
    shortcut = tf.layers.batch_normalization(shortcut,
                                             momentum=params.batchnorm_momentum,
                                             training=is_training,
                                             name=name + "_bn_short")

    feat = feat + shortcut
    feat = relu(feat, name=name + '_relu_final')
    return feat


def identity_block(input_features, kernel_sizes, filters, params, is_training, relu, name):
    """ ResNet Conv Block with shortcut. There are no strides so the shortcut does not need an extra conv.

    Args:
        input_features:
        kernel_sizes:
        filters:
        params:
        is_training:
        relu:
        name:
    :return:
    """
    n_comp = len(kernel_sizes)
    assert(n_comp == len(filters) and (n_comp == 2 or n_comp == 3))
    feat = tf.layers.conv1d(input_features,
                            filters[0],
                            kernel_sizes[0],
                            padding='same',
                            activation=None,
                            use_bias=False,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                            name=name + "_conv0")
    feat = tf.layers.batch_normalization(feat,
                                         momentum=params.batchnorm_momentum,
                                         training=is_training,
                                         name=name + "_bn0")
    feat = relu(feat, name=name + '_relu0')

    feat = tf.layers.conv1d(feat,
                            filters[1],
                            kernel_sizes[1],
                            padding='same',
                            activation=None,
                            use_bias=False,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                            name=name+"_conv1")
    feat = tf.layers.batch_normalization(feat,
                                         momentum=params.batchnorm_momentum,
                                         training=is_training,
                                         name=name + "_bn1")

    if n_comp == 3:
        feat = relu(feat, name=name + '_relu1')
        feat = tf.layers.conv1d(feat,
                                filters[2],
                                kernel_sizes[2],
                                padding='same',
                                activation=None,
                                use_bias=False,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                name=name+"_conv2")
        feat = tf.layers.batch_normalization(feat,
                                             momentum=params.batchnorm_momentum,
                                             training=is_training,
                                             name=name + "_bn2")

    feat = feat + input_features
    feat = relu(feat, name=name + '_relu_final')
    return feat


def resnet_1D(features, params, is_training=None, reuse_variables=None, aux_features=None):
    """ Build a ResNet.
        Modified ResNet-18, the blocks are: [3/64, 3/64], [3/128, 3/128], [3/256, 3/256], [3/512, 3/512]
        The default number of blocks: [2, 2, 2, 2]
        The last 3 blocks can downsample the features.
        N fully-connected layers are appended to the output the res blocks.
        There are actually 2 more layers than standard ResNet implementation.

        The downsample in ResNet-50 with 1*1 kernel may lose the frequency resolution.

        About the network parameters (no batchnorm included):
        TDNN: 2.6M (or 4.2M without dilation)
        ETDNN: 4.4M (or 7.6M without dilation)
        Modified FTDNN: 9.2M
        Modified EFTDNN: 32M
        FTDNN: 9.0M
        EFTDNN: 19.8M (much smaller than modified eftdnn)

        ResNet-18: 13.5M
        ResNet-34: 23.6M
        ResNet-50: 16.1M
        ResNet-101: 28.4M

        Args:
            features: A tensor with shape [batch, length, dim].
            params: Configuration loaded from a JSON.
            is_training: True if the network is used for training.
            reuse_variables: True if the network has been built and enable variable reuse.
            aux_features: Auxiliary features (e.g. linguistic features or bottleneck features).
        :return:
            features: The output of the last layer.
            endpoints: An OrderedDict containing output of every components. The outputs are in the order that they add to
                       the network. Thus it is convenient to split the network by a output name
    """
    # The strides only affect the last 3 conv block
    time_stride = 2 if params.resnet_time_stride else 1

    # The dimension of the features should be 40
    #assert(shape_list(features)[-1] == 40)

    tf.logging.info("Build a ResNet_1D network.")
    # ReLU is a normal choice while other activation function is possible.
    relu = tf.nn.relu
    if "network_relu_type" in params.dict:
        if params.network_relu_type == "prelu":
            relu = prelu
        elif params.network_relu_type == "lrelu":
            relu = tf.nn.leaky_relu
        elif params.network_relu_type == 'elu':
            relu = tf.nn.elu

    # The block parameters
    # default: [2, 2, 2, 2]
    if "resnet_blocks" not in params.dict:
        params.dict["resnet_blocks"] = [2, 2, 2, 2]
    tf.logging.info("The resnet blocks: [%d, %d, %d, %d]",
                    params.resnet_blocks[0], params.resnet_blocks[1], params.resnet_blocks[2], params.resnet_blocks[3])

    endpoints = OrderedDict()
    with tf.variable_scope("resnet_1D", reuse=reuse_variables):
        # features: [batch_size, lenght, dim]: [11, 300, 23]
        # ndim = shape_list(features)[-1]

        # First conv
        # No strides are applied.
        features = tf.layers.conv1d(features,
                                    16,
                                    (3),
                                    padding='same',
                                    activation=None,
                                    use_bias=False,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                    name='conv0_1')
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="conv0_bn")
        features = relu(features, name='conv0_relu')
        if params.resnet_maxpooling:
            features = tf.layers.max_pooling1d(features, (3), (1, 1), padding='same', name='conv0_max')

        # Conv Block 1
        features = conv_block(features, [[1], [3], [1]], [16, 16, 64], [1], params, is_training, relu, "conv1a")
        for i in range(params.resnet_blocks[0] - 1):
            features = identity_block(features, [[1], [3], [1]], [16, 16, 64], params, is_training, relu, "conv1b_%d" % i)

        # Conv Block 2
        features = conv_block(features, [[1], [3], [1]], [32, 32, 128], [time_stride], params, is_training, relu, "conv2a")
        for i in range(params.resnet_blocks[1] - 1):
            features = identity_block(features, [[1], [3], [1]], [32, 32, 128], params, is_training, relu, "conv2b_%d" % i)

        # Conv Block 3
        features = conv_block(features, [[1], [3], [1]], [64, 64, 256], [time_stride], params, is_training, relu, "conv3a")
        for i in range(params.resnet_blocks[2] - 1):
            features = identity_block(features, [[1], [3], [1]], [64, 64, 256], params, is_training, relu, "conv3b_%d" % i)

        # Conv Block 4
        features = conv_block(features, [[1], [3], [1]], [128, 128, 512], [time_stride], params, is_training, relu, "conv4a")
        for i in range(params.resnet_blocks[3] - 1):
            features = identity_block(features, [[1], [3], [1]], [128, 128, 512], params, is_training, relu, "conv4b_%d" % i)
        endpoints['resnet_output'] = features
        
        # Attentive Statistics Pooling
        
        features = self_attention(features, aux_features, endpoints, params, is_training)
         
#        # features: [N, L/t, 5, 512]
#        # The original resnet use average pooling to get [N, 512] which I think will eliminate the time resolution.
#        # Hence, in this implementation, we first obtain [N, L, 512] via conv layer and use dense layer to process the features
#        features = tf.layers.conv1d(features,
#                                    512,
#                                    (shape_list(features)[1]),
#                                    activation=None,
#                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
#                                    name='conv5')
#
#
#        features = tf.layers.batch_normalization(features,
#                                                 momentum=params.batchnorm_momentum,
#                                                 training=is_training,
#                                                 name="conv5_bn")
#        features = relu(features, name='conv5_relu')
#        features = tf.squeeze(features, axis=2)

        # FC layers * 2
        # Utterance-level network
        # Layer 6: [b, 512]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name='dense1')
        endpoints['dense1'] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="dense1_bn")
        features = relu(features, name='dense1_relu')
        endpoints['dense1_relu'] = features
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name='dense2')
        endpoints['dense2'] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="dense2_bn")
        features = relu(features, name='dense2_relu')
        endpoints['dense2_relu'] = features
        # Compute the number of parameters
        num_params = 3*3*64 + (2*3*3*64*64*params.resnet_blocks[0] + 64*64) + \
                     (3*3*64*128 + 3*3*128*128 + 64*128 + 2*3*3*128*128*(params.resnet_blocks[1]-1)) + \
                     (3*3*128*256 + 3*3*256*256 + 128*256 + 2*3*3*256*256*(params.resnet_blocks[2]-1)) + \
                     (3*3*256*512 + 3*3*512*512 + 256*512 + 2*3*3*512*512*(params.resnet_blocks[3]-1)) + \
                     (1*5*512*512 + 512*512 + 512*1500)
        tf.logging.info("The number of parameters of the frame-level network: %d" % num_params)
        num_layers = 4 + 2 * (params.resnet_blocks[0] + params.resnet_blocks[1] + params.resnet_blocks[2] + params.resnet_blocks[3])
        tf.logging.info("The number of layers: %d" % num_layers)

        # Layer 7: [b, x]
        if "num_nodes_last_layer" not in params.dict:
            # The default number of nodes in the last layer
            params.dict["num_nodes_last_layer"] = 512

        features = tf.layers.dense(features,
                                   params.num_nodes_last_layer,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name='tdnn7_dense')
        endpoints['tdnn7_dense'] = features

        if "last_layer_no_bn" not in params.dict:
            params.last_layer_no_bn = False

        if not params.last_layer_no_bn:
            features = tf.layers.batch_normalization(features,
                                                     momentum=params.batchnorm_momentum,
                                                     training=is_training,
                                                     name="tdnn7_bn")
            endpoints["tdnn7_bn"] = features

        if "last_layer_linear" not in params.dict:
            params.last_layer_linear = False

        if not params.last_layer_linear:
            # If the last layer is linear, no further activation is needed.
            features = relu(features, name='tdnn7_relu')
            endpoints["tdnn7_relu"] = features

    return features, endpoints

if __name__ == "__main__":

    from misc.utils import ParamsPlain
    params = ParamsPlain()
    params.dict["weight_l2_regularizer"] = 1e-5
    params.dict["batchnorm_momentum"] = 0.99
    params.dict["pooling_type"] = "statistics_pooling"
    params.dict["last_layer_linear"] = False
    params.dict["output_weight_l2_regularizer"] = 1e-4
    params.dict["network_relu_type"] = "prelu"

    # If the norm (s) is too large, after applying the margin, the softmax value would be extremely small
    params.dict["asoftmax_lambda_min"] = 10
    params.dict["asoftmax_lambda_base"] = 1000
    params.dict["asoftmax_lambda_gamma"] = 1
    params.dict["asoftmax_lambda_power"] = 4

    params.dict["amsoftmax_lambda_min"] = 10
    params.dict["amsoftmax_lambda_base"] = 1000
    params.dict["amsoftmax_lambda_gamma"] = 1
    params.dict["amsoftmax_lambda_power"] = 4
    params.dict["resnet_time_stride"] = True
    params.dict["arcsoftmax_lambda_min"] = 10
    params.dict["arcsoftmax_lambda_base"] = 1000
    params.dict["arcsoftmax_lambda_gamma"] = 1
    params.dict["arcsoftmax_lambda_power"] = 4
    params.dict["resnet_maxpooling"] = False
    params.dict["feature_norm"] = True
    params.dict["feature_scaling_factor"] = 20

    params.dict["att_value_input"] = "resnet_output"
    params.dict["att_key_input"] = "resnet_output"
    params.dict["att_key_num_nodes"] = [512]
    params.dict["att_key_network_type"] = 1
    params.dict["att_value_num_nodes"] = []
    params.dict["att_num_heads"] = 1
    params.dict["att_split_value"] = False
    params.dict["att_split_key"] = False
    params.dict["att_use_scale"] = False
    params.dict["att_apply_nonlinear"] = False
    params.dict["att_penalty_term"] = 0
    params.dict["att_key_network_type"] = 2

    batch_size = 11
    lenght, dim = 200, 23
    inputs = tf.random_uniform((batch_size, lenght, dim))

    features, endpoints = resnet_1D(inputs, params, 1)
    print(features.shape)
    #for k in endpoints:
    #    print(endpoints[k])
