# Semantic Segmentation
## Introduction
In this project, I implemented the algorithm to label the pixels of a road in images using 
a Fully Convolutional Network (FCN). The algorithm is based on 
[FCN-8 architecture](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

The source code can be downloaded from [here](https://github.com/fwa785/CarND-Semantic-Segmentation).

## The Model

The network consists of two parts: Encoder and Decoder. 

### Encoder

The encoder is a pre-trained
VGG16 classification CNN. 
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

### Decoder
The decoder takes output from layer7, does 1x1 convolution to preserve the spatial 
information and convert the channels to num_classes, which is 2.

    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='SAME',
                kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
 
Next unsample the layer after 1x1 convolution by 2x with transpose convolution operation with kernel size
4 and stride 2
    layer7_unsample = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, strides=(2, 2),  padding='SAME',
                    kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

Then the output from layer4 is converted with 1x1 convolution so it matches the size of the 1x1
convolution for layer 7, and it can add on high spatial resolution to the feature maps
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1),  padding='SAME',
                kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    output = tf.add(layer7_unsample, layer4_1x1)

Next unsample the added layer by 2x with transpose convolution to map the size of layer 3 output
    layer4_unsample = tf.layers.conv2d_transpose(output, num_classes, 4, strides=(2, 2),  padding='SAME',
                kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

Then the output of layer 3 is converted with 1x1 convolution, and added on top of 2x layer4 + 4x layer 7 
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1),  padding='SAME',
                kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))   
    output = tf.add(layer4_unsample, layer3_1x1)

Finally the output is unsampled with kernel = 16 and stride = 8 to get 4x output to match the original image
size

    output = tf.layers.conv2d_transpose(output, num_classes, 16, strides=(8, 8),  padding='SAME',
                kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))   

With the above FCN layers, the model can do semantic segmentation of any size image and identify each pixel
of the image whether it's road or not.

### Optimizer

A loss function is defined use the reduced mean of cross entropy, and Adam optimizer is used as the optimization 
algorithm.

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)


### Training Function

The training function has hyper parameters such as epoches, batch_size, keep_prob and learning rate as input.
It went through multiple epoches, and each time loads batch of image and correct label for the image. The total loss
of the each epoch is printed.

    for i in range(epochs):
        total_loss = 0
        for image, label in get_batches_fn(batch_size):           
            _, loss = sess.run([train_op, cross_entropy_loss], 
            feed_dict={input_image: image, correct_label: label, 
            keep_prob: 0.5, learning_rate: 0.001})
            total_loss = total_loss + loss
        print("EPOCH {} ...".format(i+1))
        print("Loss = {:.3f}".format(total_loss))


# Result

This is one example of the output image with epoches=20 and the batch_size=16 

![sample out image](file://runs/1545894699.054873/um_000006.png)


## Setup
### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

## Run
Run the following command to run the project:
```
python main.py
```
