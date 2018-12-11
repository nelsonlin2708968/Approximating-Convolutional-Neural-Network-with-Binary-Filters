#!/usr/bin/env/ python
# ECBM E4040 Fall 2018 Assignment 2
# TensorFlow CNN example

import tensorflow as tf
import numpy as np
import time


# Defing various layersni
# Add another layer
# Using image generator

class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed, index = 0, stride = 1):
        """
        :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
        :param in_channel: The 4-th dimension (channel number) of input matrix. For example, in_channel = 3 means the input 
        contains 3 channels.
        :param out_channel: The 4-th dimension (channel number) of output matrix. For example, out_channel=5 means the output 
        contains 5 channels (feature maps).
        :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param index: The index of the layer. It is used for naming only.
        :param stride: The stride used for convolution
        """
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            # Convolution kernel
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name = 'conv_kernel_%d' % index, shape = w_shape,
                                         initializer = tf.glorot_uniform_initializer(seed = rand_seed))
                self.weight = weight
            # Convolution bias
            with tf.name_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape = b_shape,
                                       initializer = tf.glorot_uniform_initializer(seed = rand_seed))
                self.bias = bias

            # Convolution output
            conv_out = tf.nn.conv2d(input_x, weight, strides = [1, stride, stride, 1], padding = "SAME")
            cell_out = conv_out + bias
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class approx_conv(object):
    def __init__(self, num_binary_filters, convolution_filters, convolution_biases, index = 0, stride = 1):
        """
        :param num_binary_filters: number of binary filters we want to use to approximate the weights
        :param convolution_filters: saved weights from trained full precision filters
        :param convotuion_biases: saved bias from the full precision convolutional filters
        :param index: The index of the layer. It is used for naming only.
        :param stride: The stride used for convolution
        """
        
        with tf.variable_scope('approx_conv_%d' % index):

            # Initializing convolutional kernel and bias  variables from full weight convolution filters and convolution biases
            with tf.name_scope('approx_conv_kernel'):
                filters = tf.get_variable(name='approx_conv_kernel_%d' % index, initializer = convolution_filters)
                self.weight = filters
            with tf.name_scope('approx_conv_bias'):
                biases = tf.get_variable(name='approx_conv_bias_%d' % index, initializer = convolution_biases)
                self.biases = biases

            # Creating binary filters
            binary_filters = compute_binary_filters(filters, num_binary_filters)

            # Getting alphas after fixing binary filters
            alphas, alphas_training, alphas_loss = train_alphas(filters, binary_filters, num_binary_filters)
            self.alphas = alphas
            self.alphas_training = alphas_training
            self.alphas_loss = alphas_loss
        
            dim_number = (len(filters.get_shape()))
            
            # Defining layer object within the approx_conv object
            class approx_conv_layer(object):
                def __init__(self, input_x, index = index):
                    with tf.variable_scope('approx_conv_layer_%d' % index):
                        #convolute approximations calculations: ouput = sum(alpha * conv(B, input))
                        approx_conv_outputs = []
                        
                        # reshaping alphas to multiply with conv_outputs
                        reshaped_alphas = tf.reshape(alphas, shape = [num_binary_filters] + [1] * dim_number)
                       
                        # getting conv(B, input_x) for every binary filters
                        for i in range(num_binary_filters):
                            conv_b_a = tf.nn.conv2d(input_x, binary_filters[i], strides = [1, stride, stride, 1], 
                                                    padding = "SAME")
                            approx_conv_outputs.append(conv_b_a + biases)
                            
                        conv_outputs = tf.convert_to_tensor(approx_conv_outputs, dtype = tf.float32)

                        # Sum binary convolutions * alpha
                        cell_out = tf.reduce_sum(tf.multiply(conv_outputs, reshaped_alphas), axis = 0)
                        self.cell_out = cell_out

                def output(self):
                    return self.cell_out
                
            self.layer = approx_conv_layer
            
            
class fc_layer(object):
    def __init__(self, input_x, in_size = None, out_size = None, weight_initializer = None, bias_initializer = None, 
                 rand_seed = 255,  index = 0):
        """
        :param input_x: The input of the FC layer. It should be a flatten vector.
        :param in_size: The length of input vector.
        :param out_size: The length of output vector.
        :param weight_initializer : Weight initializer if we want to use pretrained model 
        :param bias_initializer : Bias initializer if we want to use pretrained model
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param index: The index of the layer. It is used for naming only.

        """
       # Defining layer and kernel for fully connect layer. We can initialize them with given inputs.
        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                if weight_initializer is not None:
                    weight = tf.get_variable(name ='fc_kernel_%d' % index, initializer = weight_initializer)
                else:
                    w_shape = [in_size, out_size]
                    weight = tf.get_variable(name ='fc_kernel_%d' % index, shape = w_shape, 
                                             initializer = tf.glorot_uniform_initializer(seed = rand_seed))
                self.weight = weight

            with tf.name_scope('fc_bias'):
                if bias_initializer is not None:
                    bias = tf.get_variable(name='fc_bias_%d' % index, initializer = bias_initializer)
                else:
                    b_shape = [out_size]
                    bias = tf.get_variable(name='fc_bias_%d' % index, shape = b_shape,
                                           initializer = tf.glorot_uniform_initializer(seed = rand_seed))
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out
            
            
class activation_layer(object):
    def __init__(self, input_x):
    
        """
        :param input_x: The input of the activation layer.
        
        """
        cell_out = tf.nn.relu(input_x)
        self.cell_out = cell_out
        
    def output(self):
        return self.cell_out

    
class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding = "VALID"):
        """
        :param input_x: The input of the pooling layer.
        :param k_size: The kernel size you want to behave pooling action.
        :param padding: The padding setting. Read documents of tf.nn.max_pool for more information.
        """
        with tf.variable_scope('max_pooling'):
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides = pooling_shape,
                                      ksize = pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class norm_layer(object):
    def __init__(self, input_x, is_training):
        """
        :param input_x: The input that needed for normalization.
        :param is_training: To control the training or inference phase
        """
        with tf.variable_scope('batch_norm'):
            batch_mean, batch_variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
            ema = tf.train.ExponentialMovingAverage(decay=0.99)

            def True_fn():
                ema_op = ema.apply([batch_mean, batch_variance])
                with tf.control_dependencies([ema_op]):
                    return tf.identity(batch_mean), tf.identity(batch_variance)

            def False_fn():
                return ema.average(batch_mean), ema.average(batch_variance)

            mean, variance = tf.cond(is_training, True_fn, False_fn)

            cell_out = tf.nn.batch_normalization(input_x,
                                                 mean,
                                                 variance,
                                                 offset=None,
                                                 scale=None,
                                                 variance_epsilon=1e-6,
                                                 name=None)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

class dropout_layer(object):
    def __init__(self, input_x, keep_prob = 0.8):
        """
        :param input_x: The input of the dropout layer.
        :param keed_prob: the rate at which we keep weights.
        """
        with tf.variable_scope('dropout'):
            cell_out = tf.nn.dropout(input_x, keep_prob=keep_prob)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out
    
    
    
# Defining our net structure
def our_net(input_x, input_y, is_training, is_binary, pretrained_model = {}, num_binary_filters = 5,
            channel_num = 3, output_size = 10, conv_featmap = [32, 32, 64, 64, 96, 96], fc_units = [84, 84],
            conv_kernel_size = [5, 5, 3, 3, 3, 3], pooling_size = [2, 2, 2, 2, 2, 2], seed = 235, batch_size = 32):
    """
        We build our NN with similar structure as in the paper, with maxpool before activation.


        :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
        :param input_y : The labels associated to input_x
        :param is_training: To control the training or inference phase
        :param is_binary: False if we want a full-weight cnn, True if we want an approximated cnn using binary filters.
        :param pretrained_model : a dictionary with weights and biases of pretrained model. Only useful if is_binary = True.
        :param num_binary_filters : Number of binary filters to approximate full-weight cnn. Only useful if is_binary = True.
        :param channel_num: The 4-th dimension (channel number) of input matrix. For example, channel_num = 3 means the input 
         contains 3 channels. Only useful if is_binary = False.
        :param output_size : The size of our output (number of different labels)
        :param conv_featmap : The convolution map of our convolutional layers. Only useful if is_binary = False.
        :param fc_units : The number of units for fully connected layers. Only useful if is_binary = False.
        :param conv_kernel_size: the size of the cnn kernels. Only useful if is_binary = False.
        :param pooling_size: The kernel size you want to behave pooling action.
        :param rand_seed : The random seed used to generate the initial parameter value. Only useful if is_binary = False.
        
        returns :
        :fc_norm_layer_1.output() : The output of the last layer
        :cross_entropy_loss : The cross entropy loss
        :alphas_training_operations : The list of operations needed to train alphas. Empty if is_binary = False.

    """

    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)
    
    # if the net has binary weights, save the alpha training operations and create the approximate convolution layers using   
    # the initial weights of the fully trained one 
    alphas_training_operations = []
    if is_binary :
        #create convolutional objects    
        approx_conv_0 = approx_conv(num_binary_filters, pretrained_model["conv_w0"], pretrained_model["conv_b0"], index = 0)
        approx_conv_1 = approx_conv(num_binary_filters, pretrained_model["conv_w1"], pretrained_model["conv_b1"], index = 1)
        approx_conv_2 = approx_conv(num_binary_filters, pretrained_model["conv_w2"], pretrained_model["conv_b2"], index = 2)
        approx_conv_3 = approx_conv(num_binary_filters, pretrained_model["conv_w3"], pretrained_model["conv_b3"], index = 3)
        approx_conv_4 = approx_conv(num_binary_filters, pretrained_model["conv_w4"], pretrained_model["conv_b4"], index = 4)
        approx_conv_5 = approx_conv(num_binary_filters, pretrained_model["conv_w5"], pretrained_model["conv_b5"], index = 5)

        #add alphas of first conv layer to list of alphas operations to be trained
        alphas_training_operations.append(approx_conv_0.alphas_training) 
        alphas_training_operations.append(approx_conv_1.alphas_training)
        alphas_training_operations.append(approx_conv_2.alphas_training)
        alphas_training_operations.append(approx_conv_3.alphas_training) 
        alphas_training_operations.append(approx_conv_4.alphas_training)
        alphas_training_operations.append(approx_conv_5.alphas_training)

    # conv layer 0
    if is_binary :
        conv_layer_0 = approx_conv_0.layer(input_x = input_x)
        
    else :
        conv_layer_0 = conv_layer(input_x = input_x,
                                  in_channel = channel_num,
                                  out_channel = conv_featmap[0],
                                  kernel_shape = conv_kernel_size[0],
                                  rand_seed = seed, index = 0)
    conv_norm_layer_0 = norm_layer(input_x = conv_layer_0.output(),is_training=is_training)
    
    # activation layer 0
    conv_act_layer_0 = activation_layer(input_x = conv_norm_layer_0.output())
    
    # conv layer 1
    if is_binary :
        conv_layer_1 = approx_conv_1.layer(input_x = conv_act_layer_0.output())
    else :
        conv_layer_1 = conv_layer(input_x = conv_act_layer_0.output(),
                                  in_channel = conv_featmap[0],
                                  out_channel = conv_featmap[1],
                                  kernel_shape = conv_kernel_size[1],
                                  rand_seed = seed, index = 1)
    # pooling layer 1
    pooling_layer_1 = max_pooling_layer(input_x = conv_layer_1.output(),
                                        k_size=pooling_size[1])
    # batch norm layer 1
    conv_norm_layer_1 = norm_layer(input_x = pooling_layer_1.output(),is_training=is_training)
    
    # activation layer 1
    conv_act_layer_1 = activation_layer(input_x = conv_norm_layer_1.output())
    
    # dropout layer 1
    conv_dropout_layer_1 = dropout_layer(input_x = conv_act_layer_1.output())

    # conv layer 2
    if is_binary :
        conv_layer_2 = approx_conv_2.layer(input_x = conv_dropout_layer_1.output())
    else :
        conv_layer_2 = conv_layer(input_x = conv_dropout_layer_1.output(),
                                  in_channel = conv_featmap[1],
                                  out_channel = conv_featmap[2],
                                  kernel_shape = conv_kernel_size[2],
                                  rand_seed = seed, index = 2)
    # batch norm layer 2
    conv_norm_layer_2 = norm_layer(input_x = conv_layer_2.output(),is_training=is_training)
    
    # activation layer 2
    conv_act_layer_2 = activation_layer(input_x = conv_norm_layer_2.output())
    
    # conv layer 3
    if is_binary :
        conv_layer_3 = approx_conv_3.layer(input_x = conv_act_layer_2.output())
    else :
        conv_layer_3 = conv_layer(input_x = conv_act_layer_2.output(),
                                  in_channel = conv_featmap[2],
                                  out_channel = conv_featmap[3],
                                  kernel_shape = conv_kernel_size[3],
                                  rand_seed = seed, index = 3)
        
    # pooling layer 3
    pooling_layer_3 = max_pooling_layer(input_x = conv_layer_3.output(),
                                        k_size=pooling_size[3])
    
    # batch norm layer 3
    conv_norm_layer_3 = norm_layer(input_x = pooling_layer_3.output(),is_training=is_training)
    
    # activation layer 3
    conv_act_layer_3 = activation_layer(input_x = conv_norm_layer_3.output())
    
    # dropout layer 3
    conv_dropout_layer_3 = dropout_layer(input_x = conv_act_layer_3.output())
    
    # conv layer 4
    if is_binary :
        conv_layer_4 = approx_conv_4.layer(input_x = conv_dropout_layer_3.output())
    else :
        conv_layer_4 = conv_layer(input_x = conv_dropout_layer_3.output(),
                                  in_channel = conv_featmap[3],
                                  out_channel = conv_featmap[4],
                                  kernel_shape = conv_kernel_size[4],
                                  rand_seed = seed, index = 4)
    # batch norm layer 4
    conv_norm_layer_4 = norm_layer(input_x = conv_layer_4.output(),is_training=is_training)
    
    #activation layer 4
    conv_act_layer_4 = activation_layer(input_x = conv_norm_layer_4.output())
    
    # conv layer 5
    if is_binary :
        conv_layer_5 = approx_conv_5.layer(input_x = conv_act_layer_4.output())
    else :
        conv_layer_5 = conv_layer(input_x = conv_act_layer_4.output(),
                                  in_channel = conv_featmap[4],
                                  out_channel = conv_featmap[5],
                                  kernel_shape = conv_kernel_size[5],
                                  rand_seed = seed, index = 5)
        
    # pooling layer 5
    pooling_layer_5 = max_pooling_layer(input_x = conv_layer_5.output(),
                                        k_size=pooling_size[5])
    
    # batch norm layer 5
    conv_norm_layer_5 = norm_layer(input_x = pooling_layer_5.output(),is_training=is_training)
    
    # activation layer 5
    conv_act_layer_5 = activation_layer(input_x = conv_norm_layer_5.output())
    
    # dropout layer 5
    conv_dropout_layer_5 = dropout_layer(input_x = conv_act_layer_5.output())


    # flatten
    act_shape = conv_dropout_layer_5.output().get_shape()
    img_vector_length = act_shape[1].value * act_shape[2].value * act_shape[3].value
    flatten = tf.reshape(conv_dropout_layer_5.output(), shape = [-1, img_vector_length])
    
    # fc layer 0
    if is_binary :
        
        fc_layer_0 = fc_layer(input_x = flatten,
                      weight_initializer = pretrained_model["fc_w0"],
                      bias_initializer = pretrained_model["fc_b0"],
                      rand_seed = seed,
                      index=0)
    else :
        fc_layer_0 = fc_layer(input_x = flatten,
                              in_size = img_vector_length,
                              out_size = fc_units[0],
                              rand_seed = seed,
                              index=0)
    # fc activation layer 0
    fc_act_layer_0 = activation_layer(input_x = fc_layer_0.output())
    
    # fc batchnorm layer 0
    fc_norm_layer_0 = norm_layer(input_x = fc_act_layer_0.output(),is_training=is_training)
    
    # fc dropout layer 0
    fc_dropout_layer_0 = dropout_layer(input_x = fc_act_layer_0.output(), keep_prob = .5)
    
    # fc layer 1
    
    
    if is_binary :
        fc_layer_1 = fc_layer(input_x = fc_dropout_layer_0.output(),
                              weight_initializer = pretrained_model["fc_w1"],
                              bias_initializer = pretrained_model["fc_b1"],
                              rand_seed = seed,
                              index=1)
    else :
        fc_layer_1 = fc_layer(input_x = fc_dropout_layer_0.output(),
                              in_size = fc_units[1],
                              out_size = output_size,
                              rand_seed = seed,
                              index = 1)
        
    # fc activation layer 1
    fc_act_layer_1 = activation_layer(input_x = fc_layer_1.output())
    
    #fc batch norm layer 1
    fc_norm_layer_1 = norm_layer(input_x = fc_act_layer_1.output(), is_training=is_training)


    # loss
    with tf.name_scope("loss"):
        
        label = tf.one_hot(input_y, 10)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels = label, logits = fc_norm_layer_1.output()),
            name='cross_entropy')

        tf.summary.scalar('our_loss', cross_entropy_loss)

    return fc_norm_layer_1.output(), cross_entropy_loss, alphas_training_operations   



#------- useful functions for training alphas, weights and biases and getting binary filters

    
def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        with tf.get_default_graph().gradient_override_map({"Sign": "Identity"}):
            step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return step


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y)
        tf.summary.scalar('our_net_error_num', error_num)
    return error_num

    
def compute_binary_filters(convolution_filters, num_binary_filters):
    
    """
    :param convolution_filters: The full weight convolution filters tht we want to approximate.
    :param num_binary_filters: The number of binary filters we want to approximate
    
    return binarized_filters : The binarized filters used for approximation
    """
    with tf.name_scope("compute_binary_filters"):
        
        # Computer shifted standard deviation to get our base of binary filters
        dim_axes = list(range(len(convolution_filters.get_shape())))
        dim_number = len(convolution_filters.get_shape())
        mean, var = tf.nn.moments(convolution_filters, axes = dim_axes)
        stddev = tf.sqrt(var)
        
        
        if num_binary_filters == 1:
            spreaded_deviation = tf.convert_to_tensor([0], dtype=tf.float32)
        else : 
            spreaded_deviation = -1. + (2./(num_binary_filters - 1)) * tf.convert_to_tensor(list(range(num_binary_filters)), 
                                                                                            dtype=tf.float32)
        
        shifted_stddev = spreaded_deviation * stddev
        
        # Normalize the filters by removing average 
        mean_adjusted_filters = convolution_filters - mean
        
        # tile filters together
        expand_filters = tf.expand_dims(mean_adjusted_filters, axis=0)
        tiled_filters = tf.tile(expand_filters, [num_binary_filters] + [1] * dim_number)
        
        expand_stddev = tf.reshape(shifted_stddev, [num_binary_filters] + [1] * dim_number)
        
        #final binarized filters
        binarized_filters = tf.sign(tiled_filters + expand_stddev)
        
        return binarized_filters
    

def train_alphas(convolution_filters, binary_filters, num_binary_filters):
    with tf.name_scope("train_alphas"):
        
        # reshaping filters
        
        reshaped_convolution_filters = tf.reshape(convolution_filters, [-1])
        reshaped_binary_filters = tf.reshape(binary_filters, [num_binary_filters, -1])
        
        # Creating variable for alphas. 
        alphas = tf.Variable(tf.random_normal(shape=(num_binary_filters, 1), mean=1.0, stddev=0.1), name = 'alphas')
        
        # Calculating weighted sum B*alpha
        weighted_sum = tf.reduce_sum(tf.multiply(alphas, reshaped_binary_filters), axis=0)
        
        # Calculating loss and error between W and B*alpha
        error = tf.square(reshaped_convolution_filters - weighted_sum)
        loss = tf.reduce_mean(error, axis=0)
        
        # training alphas using adam optimizer
        alpha_training = tf.train.AdamOptimizer().minimize(loss, var_list=[alphas])
        
        return alphas, alpha_training, loss

    
def get_batches(x, y, batch_size, shuffle = False):
    '''
    Partition data array into mini-batches
    input:
    x: input features
    y: input labels
    output:
    x: inputs
    y: targets   
    '''
    N = y.shape[0]
    batch_count = 0
    n_batches = N//batch_size
    while True:
        if(batch_count < n_batches):
            yield (x[batch_count * batch_size : (batch_count +1) * batch_size], 
                   y[batch_count * batch_size : (batch_count +1) * batch_size]) # yield new batch
            batch_count += 1
        else:
            if shuffle == True:
                    p = np.random.permutation(N) #create random array of indexes
                    x = x[p]
                    y = y[p]
            batch_count = 0

        
# training function
def training(X_train, y_train, X_val, y_val, is_binary,
             pretrained_model = {},
             num_binary_filters = 5,
             conv_featmap = [6],
             fc_units = [84],
             conv_kernel_size = [5],
             pooling_size = [2],
             seed = 235,
             learning_rate = 1e-2,
             epoch = 20,
             alpha_training_epochs = 200,
             batch_size = 245,
             verbose = True):

    # define the variables and parameter needed during training
    with tf.get_default_graph().gradient_override_map({"Sign": "Identity"}):
        with tf.name_scope('inputs'):
            xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
            ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
            is_training = tf.placeholder(tf.bool, name='is_training')

        if is_binary:
            output, loss, alphas_training_operations = our_net(xs, ys, is_training, is_binary,
                                                               pretrained_model = pretrained_model,
                                                               num_binary_filters = num_binary_filters,
                                                               fc_units=fc_units,
                                                               pooling_size=pooling_size,
                                                               seed=seed,
                                                               batch_size = batch_size)
        else:
            output, loss, _ = our_net(xs, ys, is_training, is_binary,
                                      conv_featmap=conv_featmap,
                                      fc_units=fc_units,
                                      conv_kernel_size=conv_kernel_size,
                                      pooling_size=pooling_size,
                                      seed=seed,
                                      batch_size = batch_size)

        iters = int(X_train.shape[0] / batch_size)

        step = train_step(loss)
        eve = evaluate(output, ys)

        iter_total = 0
        best_acc = 0
        
        #indicates which model is being trained
        if is_binary :
            cur_model_name = 'approx_cnn_{}'.format(int(time.time()))
        else :
            cur_model_name = 'full_cnn_{}'.format(int(time.time()))
        
        #create iterator
        batch = get_batches(X_train, y_train, batch_size, shuffle = True)
        
        with tf.Session() as sess:
            merge = tf.summary.merge_all()

            writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            start_time = time.time()
            for epc in range(epoch):
                print("epoch {} ".format(epc + 1))

                for itr in range(iters):
                    iter_total += 1

                    # Training alphas in each conv layer 
                    if is_binary:
                        for alpha_training_op in alphas_training_operations:

                            for alpha_epoch in range(alpha_training_epochs):
                                sess.run(alpha_training_op)

                    training_batch_x, training_batch_y = next(batch)

                    _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x,
                                                                    ys: training_batch_y,
                                                                    is_training: True})

                    if iter_total % 100 == 0:
                        # do validation
                        valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val,
                                                                                    ys: y_val,
                                                                                    is_training: False})
                        valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                        if verbose:
                            end_time = time.time()
                            
                            print('{}/{} loss: {}, validation accuracy : {}%, {} seconds'.format(
                                batch_size * (itr + 1),
                                X_train.shape[0],
                                cur_loss,
                                valid_acc,
                                end_time - start_time))
                            start_time = end_time

                        # save the merge result summary
                        writer.add_summary(merge_result, iter_total)

                        # when achieve the best validation accuracy, we store the model paramters

                        if valid_acc > best_acc:
                            print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                            best_acc = valid_acc
                            saver.save(sess, 'model/{}'.format(cur_model_name))

            variables_to_save = {}    
            # Save weights and biases  of fully trained net to initialize the approximated net
            if is_binary == False:
                variables_to_save = {}
                variables_to_save = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                conv_w0, conv_b0, conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3,conv_w4, conv_b4,conv_w5, conv_b5, fc_w0, fc_b0, fc_w1, fc_b1 = sess.run(variables_to_save)
                variables_to_save = {"conv_w0" : conv_w0,
                                     "conv_b0" : conv_b0,
                                     "conv_w1": conv_w1,
                                     "conv_b1" : conv_b1,
                                     "conv_w2": conv_w2,
                                     "conv_b2" : conv_b2,
                                     "conv_w3" : conv_w3,
                                     "conv_b3" : conv_b3,
                                     "conv_w4": conv_w4,
                                     "conv_b4" : conv_b4,
                                     "conv_w5": conv_w5,
                                     "conv_b5" : conv_b5,
                                     "fc_w0": fc_w0,
                                     "fc_b0" : fc_b0,
                                     "fc_w1": fc_w1,
                                     "fc_b1" : fc_b1 }


    print("Training ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
    return(variables_to_save, best_acc)

