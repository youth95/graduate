from collections import OrderedDict
from models.ConvRNN import CLSTM_cell

width = 256
height = 256
# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [5, 8, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [16, 16, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [32, 32, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(width, height), input_channels=8, filter_size=5, num_features=16),
        CLSTM_cell(shape=(width // 2, height // 2), input_channels=16, filter_size=5, num_features=32),
        CLSTM_cell(shape=(width // 4, height // 4), input_channels=32, filter_size=5, num_features=32)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [32, 32, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [32, 32, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [16, 8, 3, 1, 1],
            'conv4_leaky_1': [8, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(width // 4, height // 4), input_channels=32, filter_size=5, num_features=32),
        CLSTM_cell(shape=(width // 2, height // 2), input_channels=32, filter_size=5, num_features=32),
        CLSTM_cell(shape=(width, height), input_channels=32, filter_size=5, num_features=16),
    ]
]
