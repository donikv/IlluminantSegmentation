import tensorflow as tf
import configparser
import tensorflow.keras.layers as l
from tensorflow.keras.models import Model
from general.model import CustomLayers
from ast import literal_eval
from general.model import simple_fc4


def try_parse_or_default(c, k, default):
    try:
        return try_parse(c[k])
    except:
        return default


def try_parse(v):
    try:
        return literal_eval(v)
    except:
        return v

class CustomModel(Model):

    def __init__(self, config, weights_path=None):
        super().__init__()
        self.config = config

        layer_names, named_layers, layer_input_names = self.__create_layers_from_config(self.config)
        self.layer_names = layer_names
        self.named_layers = named_layers
        self.layer_input_names = layer_input_names

        self.model = self.__build_functional_model_from_layers(layer_names, named_layers, layer_input_names)
        if weights_path != None:
            self.model.load_weights(weights_path)



    def __create_layers_from_config(self, config):
        named_layers = {}
        layer_input_names = {}
        layer_names = config.sections()
        layer_names = list(filter(lambda x: x.lower().strip() != "config", layer_names))
        layer_names = list(map(lambda x: x.strip(), layer_names))

        for i, layer_name in enumerate(layer_names):
            layer_props = config[layer_name]

            type = layer_props['type'].lower().strip()
            layer_props = dict(layer_props)

            props = {k: try_parse(v) for k, v in layer_props.items()}
            del props['type']

            try:
                inputs = try_parse(props['inputs'])
            except:
                inputs = layer_names[i-1] if i > 0 else None
            layer_input_names[layer_name] = inputs
            try:
                del props['inputs']
            except:
                pass

            props['name'] = layer_name

            if type == 'conv2d':
                layer_cls = l.Conv2D
            elif type == 'normconv2d':
                layer_cls = CustomLayers.NormConv2D
            elif type == 'maxpool':
                layer_cls = l.MaxPool2D
            elif type == 'avgpool':
                layer_cls = l.AveragePooling2D
            elif type == 'globalavgpool':
                layer_cls = l.GlobalAveragePooling2D
            elif type == 'batchnorm':
                layer_cls = l.BatchNormalization
            elif type == 'activation':
                layer_cls = l.Activation
            elif type == 'histogram':
                layer_cls = CustomLayers.DiffHist
            elif type == 'input':
                layer_cls = l.Input
            elif type == 'add':
                layer_cls = l.Add
            elif type == 'sub':
                layer_cls = l.Subtract
            elif type == 'mul':
                layer_cls = l.Multiply
            elif type == 'div':
                layer_cls = CustomLayers.Div
            elif type == 'avg':
                layer_cls = l.Average
            elif type == 'mulconstant':
                layer_cls = CustomLayers.MulConstant
            elif type == 'invert':
                layer_cls = CustomLayers.Invert
            elif type == 'dense':
                layer_cls = l.Dense
            elif type == 'dropout':
                layer_cls = l.Dropout
            elif type == 'channelslice':
                layer_cls = CustomLayers.ChannelSlice
            elif type == 'slice':
                layer_cls = CustomLayers.Slice
            elif type == 'fc4':
                layer_cls = CustomLayers.FC4
            elif type == 'masking':
                layer_cls = CustomLayers.MaskingLayer
            elif type == 'multimasking':
                layer_cls = CustomLayers.MultiMaskingLayer
            elif type == 'colorcorrection':
                layer_cls = CustomLayers.ColorCorrectionLayer
            elif type == 'cosinesegmentation':
                layer_cls = CustomLayers.CosineSegmentation
            elif type == 'rbf':
                layer_cls = CustomLayers.RBFSimilarity
            elif type == 'gwestimation':
                layer_cls = CustomLayers.GrayWorldEstimationLayer
            elif type == 'wpestimation':
                layer_cls = CustomLayers.WhitePatchEstimationLayer
            elif type == 'upsampling':
                layer_cls = l.UpSampling2D
            elif type == 'reshape':
                layer_cls = CustomLayers.Reshape
            elif type == 'resize':
                layer_cls = simple_fc4.ResizeLayer
            elif type == 'concat':
                layer_cls = CustomLayers.Concatenate
            elif type == 'index':
                layer_cls = CustomLayers.IndexLayer
            elif type == 'tile':
                layer_cls = CustomLayers.Tile
            elif type == 'branchcell':
                layer_cls = CustomLayers.BranchCell
            elif type == 'resbranch':
                layer_cls = CustomLayers.ResBranch
            elif type == 'pyconv':
                layer_cls = CustomLayers.PyConv
            elif type == 'unet':
                layer_cls = CustomLayers.Unet
            elif type == 'fpn':
                layer_cls = CustomLayers.FPN
            elif type == 'psp':
                layer_cls = CustomLayers.PSP
            elif type == 'simplefc4':
                layer_cls = simple_fc4.SimpleFc4
            elif type == 'anfc4':
                layer_cls = simple_fc4.AlexNetFC4
            elif type == 'anfc4clustering':
                layer_cls = simple_fc4.AlexNetFC4Clustering
            elif type == 'tuple':
                layer_cls = CustomLayers.Tuple
            elif type == 'normalize':
                layer_cls = CustomLayers.NormLayer
            elif type == 'brightnessnorm':
                layer_cls = CustomLayers.BrightnessNormLayer
            elif type == 'shared_weights':
                layer_cls = CustomLayers.SharedWeightsLayer
                original = named_layers[props['original_layer']]
                props['original_layer'] = original
            elif type == 'custommodel':
                cp = props['config_path']
                try:
                    weights_path = props["weights_path"]
                except:
                    weights_path = None
                mconfig = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
                mconfig.read(cp)
                for k, v in props.items():
                    mconfig['config'][k.upper()] = str(v)
                props = {'config': mconfig, "weights_path": weights_path}
                layer_cls = CustomModel
            else:
                raise ValueError(f"Unknown layer type: {type} in layer named: {layer_name}")

            if layer_cls is not None:
                layer = layer_cls(**props)
                named_layers[layer_name] = layer

        return layer_names, named_layers, layer_input_names

    def __build_functional_model_from_layers(self, layer_names, named_layers, layer_input_names):
        inpt = named_layers[layer_names[0]]
        current_outputs = {layer_names[0]:inpt}
        for layer_name in layer_names[1:]:
            layer = named_layers[layer_name]
            input_names = layer_input_names[layer_name]

            if type(input_names) is str:
                inputs = current_outputs[input_names]
            else:
                inputs = list(map(lambda x: current_outputs[x], input_names))
                # inputs = l.concatenate(inputs)

            out = layer(inputs)
            current_outputs[layer_name] = out

        model = Model(inpt, out)
        return model

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


def build_model_from_config(config):
    if type(config) is str:
        config_path = config
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(config_path)
    model = CustomModel(config)
    return model.model

if __name__ == '__main__':
    folder = 'custom_models/fc4_model3_hist'
    config_path = f'{folder}/model.config'

    model = build_model_from_config(config_path)
    model.summary()
    tf.keras.utils.plot_model(model, f"{folder}/model.png", show_shapes=True)