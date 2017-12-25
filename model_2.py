import tensorflow as tf
import os
import cPickle

class UNet_model(object):

    def __init__(self, model_id, img_height=512, img_width=1024, batch_size=4):
        self.model_id = model_id

        self.project_dir = "/n/public/lxx/"

        self.logs_dir = self.project_dir + "training_logs/"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        self.no_of_classes = 20

        self.lr = 5e-4 # (learning rate)

        # create all dirs for storing checkpoints and other log data:
        self.create_model_dirs()

        # add placeholders to the comp. graph:
        self.add_placeholders()

        # define the forward pass, compute logits and add to the comp. graph:
        self.add_logits()

        # compute the batch loss and add to the comp. graph:
        self.add_loss_op()

        # add a training operation (for minimizing the loss) to the comp. graph:
        self.add_train_op()

    def create_model_dirs(self):
        self.model_dir = self.logs_dir + "model_%s" % self.model_id + "/"
        self.checkpoints_dir = self.model_dir + "checkpoints/"
        self.debug_imgs_dir = self.model_dir + "imgs/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
            os.makedirs(self.debug_imgs_dir)

    def add_placeholders(self):
        self.imgs_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.img_height, self.img_width, 3],
                    name="imgs_ph")

        self.onehot_labels_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.img_height, self.img_width, self.no_of_classes],
                    name="onehot_labels_ph")

    def create_feed_dict(self, imgs_batch, onehot_labels_batch=None):
        # return a feed_dict mapping the placeholders to the actual input data:
        feed_dict = {}
        feed_dict[self.imgs_ph] = imgs_batch
        if onehot_labels_batch is not None:
            # only add the labels data if it's specified (during inference, we
            # won't have any labels):
            feed_dict[self.onehot_labels_ph] = onehot_labels_batch

        return feed_dict

    def add_logits(self):
        #encoder
        #input size [batch_size, w, h, 3]
        self.conv1_1 = conv(self.imgs_ph, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv1_1') # [~, w, h, 64]
        self.conv1_2 = conv(self.conv1_1, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv1_2') # [~, w, h, 64]
        self.pool1 = max_pool(self.conv1_2, 2, 2, 2, 2, padding = 'SAME', name = 'pool1') # [~, w/2, h/2, 64]
        print "conv1_1 shape: {}".format(self.conv1_1.shape)
        print "conv1_2 shape: {}".format(self.conv1_2.shape)
        print "pool1 shape: {}".format(self.pool1.shape)
        
        self.conv2_1 = conv(self.pool1, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv2_1') # [~, w/2, h/2, 128]
        self.conv2_2 = conv(self.conv2_1, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv2_2') # [~, w/2, h/2, 128]
        self.pool2 = max_pool(self.conv2_2, 2, 2, 2, 2, padding = 'SAME', name ='pool2') # [~, w/4, h/4, 128]
        print "conv2_2 shape: {}".format(self.conv2_1.shape)
        print "conv2_2 shape: {}".format(self.conv2_2.shape)
        print "pool2 shape: {}".format(self.pool2.shape)
        
        self.conv3_1 = conv(self.pool2, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv3_1') # [~, w/8, h/8, 256]
        self.conv3_2 = conv(self.conv3_1, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv3_2') # [~, w/4, h/4, 256]
        self.conv3_3 = conv(self.conv3_2, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv3_3') # [~, w/4, h/4, 256]
        self.pool3 = max_pool(self.conv3_3, 2, 2, 2, 2, padding = 'SAME', name ='pool3') # [~, w/8, h/8, 256]
        print "conv3_1 shape: {}".format(self.conv3_1.shape)
        print "conv3_2 shape: {}".format(self.conv3_2.shape)
        print "conv3_3 shape: {}".format(self.conv3_3.shape)
        print "pool3 shape: {}".format(self.pool3.shape)

        self.conv4_1 = conv(self.pool3, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv4_1') # [~, w/8, h/8, 512]
        self.conv4_2 = conv(self.conv4_1, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv4_2') # [~, w/8, h/8, 512]
        self.conv4_3 = conv(self.conv4_2, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv4_3') # [~, w/8, h/8, 512]
        self.pool4 = max_pool(self.conv4_3, 2, 2, 2, 2, padding = 'SAME', name ='pool4') # [~, w/16, h/16, 512]
        print "conv4_1 shape: {}".format(self.conv4_1.shape)
        print "conv4_2 shape: {}".format(self.conv4_2.shape)
        print "conv4_3 shape: {}".format(self.conv4_3.shape)
        print "pool4 shape: {}".format(self.pool4.shape)
        
        self.conv5_1 = conv(self.pool4, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv5_1') # [~, w/16, h/16, 512]
        self.conv5_2 = conv(self.conv5_1, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv5_2') # [~, w/16, h/16, 512]
        self.conv5_3 = conv(self.conv5_2, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv5_3') # [~, w/16, h/16, 512]
        self.pool5 = max_pool(self.conv5_3, 2, 2, 2, 2, padding = 'SAME', name = 'pool5') # [~, w/32, h/32, 512]
        print "conv5_1 shape: {}".format(self.conv5_1.shape)
        print "conv5_2 shape: {}".format(self.conv5_2.shape)
        print "conv5_3 shape: {}".format(self.conv5_3.shape)
        print "pool5 shape: {}".format(self.pool5.shape)

        #decoder
        self.deconv1 = deconv(self.pool5, 3, 3, 512, 2, 2, output_shape=[self.batch_size, self.img_height/16, self.img_width/16, 512], padding='SAME', name = 'deconv1') #[~, w/16, h/16, 512]
        self.norm1 = norm_rescale(self.pool4, 512, 'norm1')                                                                                                      #[~, w/16, h/16, 512]
        self.concat1 = tf.concat([self.deconv1, self.norm1], axis=3)                                                                                             #[~, w/16, h/16, 1024]
        print "deconv1 shape: {}".format(self.deconv1.shape)
        print "norm1 shape: {}".format(self.norm1.shape)
        print "concat1 shape: {}".format(self.concat1.shape)

        self.conv6_1 = conv(self.concat1, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv6_1') # [~, w/16, h/16, 512]
        self.conv6_2 = conv(self.conv6_1, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv6_2') # [~, w/16, h/16, 256]
        print "conv6_1 shape: {}".format(self.conv6_1.shape)
        print "conv6_2 shape: {}".format(self.conv6_2.shape)
        
        self.deconv2 = deconv(self.conv6_2, 3, 3, 256, 2, 2, output_shape=[self.batch_size, self.img_height/8, self.img_width/8, 256], padding='SAME', name = 'deconv2') #[~, w/8, h/8, 256]
        self.norm2 = norm_rescale(self.pool3, 256, 'norm2')                                                                                                      #[~, w/8, h/8, 256]
        self.concat2 = tf.concat([self.deconv2, self.norm2], axis=3)                                                                                             #[~, w/8, h/8, 512]
        print "deconv2 shape: {}".format(self.deconv2.shape)
        print "norm2 shape: {}".format(self.norm2.shape)
        print "concat2 shape: {}".format(self.concat2.shape)
        
        self.conv7_1 = conv(self.concat2, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv7_1') # [~, w/8, h/8, 256]
        self.conv7_2 = conv(self.conv7_1, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv7_2') # [~, w/8, h/8, 128]
        print "conv7_1 shape: {}".format(self.conv7_1.shape)
        print "conv7_2 shape: {}".format(self.conv7_2.shape)

        self.deconv3 = deconv(self.conv7_2, 3, 3, 128, 2, 2, output_shape=[self.batch_size, self.img_height/4, self.img_width/4, 128], padding='SAME', name = 'deconv3') #[~, w/4, h/4, 128]
        self.norm3 = norm_rescale(self.pool2, 128, 'norm3')                                                                                                      #[~, w/4, h/4, 128]
        self.concat3 = tf.concat([self.deconv3, self.norm3], axis=3)                                                                                             #[~, w/4, h/4, 256]
        print "deconv3 shape: {}".format(self.deconv3.shape)
        print "norm3 shape: {}".format(self.norm3.shape)
        print "concat3 shape: {}".format(self.concat3.shape)

        self.conv8_1 = conv(self.concat3, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv8_1') # [~, w/4, h/4, 128]
        self.conv8_2 = conv(self.conv8_1, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv8_2') # [~, w/4, h/4, 64]
        print "conv8_1 shape: {}".format(self.conv8_1.shape)
        print "conv8_2 shape: {}".format(self.conv8_2.shape)
        
        self.deconv4 = deconv(self.conv8_2, 3, 3, 64, 2, 2, output_shape=[self.batch_size, self.img_height/2, self.img_width/2, 64], padding='SAME', name = 'deconv4') #[~, w/2, h/2, 64]
        self.norm4 = norm_rescale(self.pool1, 64, 'norm4')                                                                                                     #[~, w/2, h/2, 64]
        self.concat4 = tf.concat([self.deconv4, self.norm4], axis=3)                                                                                           #[~, w/2, h/2, 128]
        print "deconv4 shape: {}".format(self.deconv4.shape)
        print "norm4 shape: {}".format(self.norm4.shape)
        print "concat4 shape: {}".format(self.concat4.shape)
        
        self.conv9_1 = conv(self.concat4, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv9_1') # [~, w/2, h/2, 64]
        print "conv9_1 shape: {}".format(self.conv9_1.shape)

        self.deconv5 = deconv(self.conv9_1, 3, 3, 64, 2, 2, output_shape=[self.batch_size, self.img_height, self.img_width, 64], padding='SAME', name = 'deconv5') #[~, w, h, 64]
        self.norm5 = norm_rescale(self.conv1_2, 64, 'norm5')
        self.concat5 = tf.concat([self.deconv5, self.norm5], axis=3)
        self.norm6 = norm_rescale(self.conv1_1, 64, 'norm6')
        self.concat6 = tf.concat([self.concat5, self.norm6], axis=3)
        self.conv10_1 = conv(self.concat6, 1, 1, self.no_of_classes, 1, 1, padding = 'SAME', non_linear="NONE", name='conv10_1')#[~, w, h, out_channels(<64)]
        print "deconv5 shape: {}".format(self.deconv5.shape)
        print "conv10_1 shape: {}".format(self.conv10_1.shape)

        self.logits = self.conv10_1

    def add_loss_op(self):

        self.loss_raw=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.onehot_labels_ph, logits=self.logits))
        self.loss = self.loss_raw    

    def add_train_op(self):
        # create the train op:
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)


"""
Predefine all necessary layer for the Model
""" 
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1, non_linear="RELU"):
  """
  Adapted from: https://github.com/ethereon/caffe-tensorflow
  """
  # Get number of input channels
  input_channels = int(x.get_shape()[-1])
  
  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k, 
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)
  
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])  
    
    
    if groups == 1:
      conv = convolve(x, weights)
      
    # In the cases of multiple groups, split inputs & weights and
    else:
      # Split input and weights and convolve them separately
      input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
      weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
      
      # Concat the convolved output together again
      conv = tf.concat(axis = 3, values = output_groups)
      
    # Add biases 
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
    # Apply non-linear function
    if non_linear == "RELU":
        nonlin = tf.nn.relu(bias, name = scope.name)
    elif non_linear == "SIGMOID":
        nonlin = tf.sigmoid(bias, name = scope.name)
    elif non_linear == 'NONE':
        nonlin = tf.identity(bias, name = scope.name)  
        
    return nonlin
  
def deconv(x, filter_height, filter_width, num_filters, stride_y, stride_x, output_shape, name,
           padding='SAME', non_linear="RELU"):

  # Get number of input channels
  input_channels = int(x.get_shape()[-1])

  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the deconv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, num_filters, input_channels])
    biases = tf.get_variable('biases', shape = [num_filters])  
    
    dconv = tf.nn.conv2d_transpose(x, weights, output_shape, [1, stride_y, stride_x, 1], padding)
    # Add biases 
    bias = tf.reshape(tf.nn.bias_add(dconv, biases), dconv.get_shape().as_list())
    # Apply non-linear function
    if non_linear == "RELU":
        nonlin = tf.nn.relu(bias, name = scope.name)
    elif non_linear == "SIGMOID":
        nonlin = tf.sigmoid(bias, name = scope.name)
    elif non_linear == "SOFTMAX":
        nonlin = tf.nn.softmax(bias, dim=-1, name = scope.name)
    elif non_linear == 'NONE':
        nonlin = tf.identity(bias, name = scope.name)  
        
    return nonlin

def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)
    
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)      
      return relu
    else:
      return act
    

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)
  
def lrn(x, radius, alpha, beta, name, bias=1.0):
  return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                            beta = beta, bias = bias, name = name)
  
def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)
  
def norm_rescale(x, channels, name):
    with tf.variable_scope(name) as scope:
        scale = tf.get_variable('scale', shape=[channels], trainable=True, dtype=tf.float32)
        return scale * tf.nn.l2_normalize(x, dim=[1, 2]) # NOTE: per feature map normalizatin
    
