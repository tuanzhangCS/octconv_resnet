from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.layers import UpSampling2D, Add, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

def oct_resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     alpha=0.5,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True,
                     oct_last=False):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): tensor or list of input tensor which from input image or previous layer,
                         the first oct-conv layer of network only has one input
                         sequense is: high-frequency, low-frequency
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (list of tensor): tensor as input to the next layer
            if oct_last is False, x contain out_h and out_l
    """
    if (not isinstance(inputs, list)):
        inputs = [inputs]

    if oct_last:
        alpha = 0

    if (not oct_last):
        convh_l = Conv2D(int(num_filters * alpha),
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))

    convh_h = Conv2D(int(num_filters * (1-alpha)),
                     kernel_size=kernel_size,
                     strides=strides,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4))
    in_h = inputs[0]

    if (len(inputs) == 2):
        if (not oct_last):
            convl_l = Conv2D(int(num_filters * alpha),
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(1e-4))

        convl_h = Conv2D(int(num_filters * (1-alpha)),
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))
        in_l = inputs[1]

    if conv_first:
        if (len(inputs) == 2):
            h0 = convh_h(in_h)
            h1 = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(in_l)
            h1 = convl_h(h1)
            out_h = Add()([h0, h1])
            if (not oct_last):
                l0 = convl_l(in_l)
                l1 = AveragePooling2D(pool_size=2)(in_h)
                l1 = convh_l(l1)
                out_l = Add()([l0, l1])
        else:
            out_h = convh_h(in_h)
            if (not oct_last):
                out_l = AveragePooling2D(pool_size=2)(in_h)
                out_l = convh_l(out_l)

        if batch_normalization:
            out_h = BatchNormalization()(out_h)
            if (not oct_last):
                out_l = BatchNormalization()(out_l)
        if activation is not None:
            out_h = Activation(activation)(out_h)
            if (not oct_last):
                out_l = Activation(activation)(out_l)
    else:
        if batch_normalization:
            in_h = BatchNormalization()(in_h)
            if (not oct_last):
                in_l = BatchNormalization()(in_l)
        if activation is not None:
            in_h = Activation(activation)(in_h)
            if (not oct_last):
                in_l = Activation(activation)(in_l)
        if (len(inputs) == 2):
            h0 = convh_h(in_h)
            h1 = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(in_l)
            h1 = convl_h(h1)
            out_h = Add()([h0, h1])
            if (not oct_last):
                l0 = convl_l(in_l)
                l1 = AveragePooling2D(pool_size=2)(in_h)
                l1 = convh_l(l1)
                out_l = Add()([l0, l1])
        else:
            out_h = convh_h(in_h)
            if (not oct_last):
                out_l = AveragePooling2D(pool_size=2)(in_h)
                out_l = convh_l(out_l)

    if (oct_last):
        return out_h
    else:
        return [out_h, out_l]


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = oct_resnet_layer(inputs=inputs) # x is list of tensor, and len(x) == 2
    # Instantiate the stack of residual units
    oct_last = False
    for stack in range(3):
        if stack == 2:
            oct_last = True
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = oct_resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             oct_last=oct_last)
            y = oct_resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             oct_last=oct_last)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims, because the size of feature map has been changed
                x = oct_resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 oct_last=oct_last)
            if oct_last:
                x = Add()([x, y])
                x = Activation('relu')(x)
            else:
                xh, xl = x
                yh, yl = y
                xh = Add()([xh, yh])
                xl = Add()([xl, yl])
                xh = Activation('relu')(xh)
                xl = Activation('relu')(xl)
                x = [xh, xl]
        num_filters *= 2


    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = oct_resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    oct_last = False
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample
                if stage == 2: # and res_block == (num_res_blocks - 1):
                    oct_last = True

            # bottleneck residual unit
            y = oct_resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False,
                             oct_last=oct_last)
            y = oct_resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False,
                             oct_last=oct_last)
            y = oct_resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False,
                             oct_last=oct_last)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = oct_resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 oct_last=oct_last)
            if oct_last:
                x = keras.layers.add([x, y])
            else:
                xh = keras.layers.add([x[0], y[0]])
                xl = keras.layers.add([x[1], y[1]])
                x = [xh, xl]

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


