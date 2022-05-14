import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Layer,InputSpec, Flatten, Conv3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate, ZeroPadding3D,Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow_addons.layers import InstanceNormalization

K_INITER="he_normal"
def G_conv3d(Nf, ks, st, pad="valid", norm=True, do_relu=True, lrelu=0, drop=0):
    """
    Conv3d step.
    `Nf`: number of output channels
    `ks`: fliter size
    `st`: stride step
    `pad`: padding style
    `norm`: Instance normalization
    `lrelu`: leaky relu rate
    `drop`: dropout rate
    """
    def func(x):
        pd = pad
        if pd == "reflect":
            if ks % 2 == 1:
                e = ks//2
                x = ReflectionPadding3D((e, e, e))(x)
                pd = "valid"
            else:
                raise NotImplementedError(
                    "fliter size must be odd if use reflect padding!")
        x = Conv3D(Nf, kernel_size=ks, strides=st,
                    kernel_initializer=K_INITER, padding=pd)(x)
        if norm:
            x = InstanceNormalization()(x)
        if do_relu:
            x = LeakyReLU(lrelu)(x)
        if drop != 0:
            x = Dropout(drop)(x)
        return x
    return func

def G_deconv3d(Nf, ks, st, pad="valid", norm=True, do_relu=True, lrelu=0, drop=0):
    """
    Deconv3d step.
    `Nf`: number of output channels
    `ks`: fliter size
    `st`: stride step
    `pad`: padding style
    `norm`: Instance normalization
    `lrelu`: leaky relu rate
    `drop`: dropout rate
    """
    def func(x):
        pd = pad
        if pd == "reflect":
            if ks % 2 == 1:
                e = ks//2
                x = ReflectionPadding3D((e, e, e))(x)
                pd = "valid"
            else:
                raise NotImplementedError(
                    "fliter size must be odd if use reflect padding!")
        x = Conv3DTranspose(
            Nf, kernel_size=ks, strides=st, kernel_initializer=K_INITER, padding=pd)(x)
        if norm:
            x = InstanceNormalization()(x)
        if do_relu:
            x = LeakyReLU(lrelu)(x)
        if drop != 0:
            x = Dropout(drop)(x)
        return x
    return func

def Reslayer(Nf, ks):
    """
    Bottom resnet layer.
    """
    def func(inputs):
        x = inputs
        x = G_conv3d(Nf, ks=ks, st=1, pad='reflect', lrelu=0.2)(x)
        x = G_conv3d(Nf, ks=ks, st=1, pad='reflect', do_relu=False)(x)

        outputs = LeakyReLU(0.2)(Add()([x,inputs]))
        return outputs
    return func

def Concatlayer(Nf, ks):
    """
    Bottom concat layer.
    """
    def func(x,last):
        x = Concatenate()([x,last])
        x = G_conv3d(Nf, ks=ks, st=1, pad='reflect',lrelu=0.2)(x)
        # x = G_conv3d(Nf, ks=ks, st=1, pad='reflect',lrelu=0.2)(x)
        return x
    return func
    
class ReflectionPadding3D(Layer):
    def __init__(self, padding, **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)
    def get_config(self):
        config=super().get_config().copy()
        config.update({
            'padding':self.padding
        })
        return config
    def compute_output_shape(self, input_shape):
        shape = (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3] + 2 * self.padding[2],
            input_shape[4]
        )
        return shape

    def call(self, x, mask=None):
        a,b,c = self.padding
        return tf.pad(
            x, [[0, 0], [a, a], [b, b], [c,c], [0, 0]],'REFLECT'
        )