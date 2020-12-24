"""
带注意力的CNN
"""
from torch import nn
from env.base import device
import torch
from models.utils import make_layers
from models.net_params import convlstm_encoder_params, convlstm_decoder_params
import pandas as pd

out_step = 8
torch.set_default_dtype(torch.float64)


class ChannelAttention(nn.Module):
    """
    通道注意力层
    """

    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    空间注意力层
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CNN(torch.nn.Module):
    """
    CNN 层
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4,
                            out_channels=8,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU()
        )
        # self.conv2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(8, 16, 3, 1, 1),
        #     torch.nn.BatchNorm2d(16),
        #     torch.nn.ReLU()
        # )
        # self.conv3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(16, 32, 3, 1, 1),
        #     torch.nn.BatchNorm2d(32),
        #     torch.nn.ReLU()
        # )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 4, 3, 1, 1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.conv4(x)

        return x


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))
        # self.l = D2NL()
        # self.CNN =CNN()

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=out_step)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        # TODO 接全链接预测位置
        # inputs = self.l(inputs)
        # inputs = self.CNN(inputs)
        return inputs


class ModelDriver(torch.nn.Module):
    def __init__(self):
        super(ModelDriver, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(4)
        self.seq2seq = Seq2Seq()
        self.cnn = CNN()

    def forward(self, x):
        batch, src_seq_len, ch, width, height = x.shape
        x = x.clone().view(-1, ch, width, height)
        _l, ch, width, height = x.shape
        # x = torch.tensor(x[:, :4, :, :]).to(device)  # 气象通道
        # _x = torch.tensor(x[:, 3:, :, :]).to(device)  # 遥感通道
        _x = x[:, 4:5, :, :]  # 遥感通道
        _label = x[:, 5:, :, :]  # 密度通道
        x = x[:, :4, :, :]
        # 过 CNN
        x = self.cnn(x)
        # print('CNN:{}'.format(x))
        # 过 注意力
        x = x * self.ca(x)
        x = x * self.sa(x)
        # print('注意力机制：{},shape:{}'.format(x, x.shape))
        rx = []
        for i in range(_l):
            _content = torch.mul(_x[i], _label[i] + 1)
            rx.append(torch.cat([x[i], _content]))
        x = torch.cat(rx)
        x = x.view(batch, src_seq_len, ch - 1, width, height)
        # 过 Seq2Seq
        x = self.seq2seq(x)
        # print('最终的x：{}'.format(x))
        return x


class Seq2Seq(torch.nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(device)
        self.decoder = Decoder(convlstm_decoder_params[0], convlstm_decoder_params[1]).to(device)

    def forward(self, inp):
        state = self.encoder(inp)
        output = self.decoder(state)
        return output


if __name__ == "__main__":
    # model = ModelDriver().to(device)
    #
    # # (batch,src_seq_len,ch,width,height)
    # x = torch.randn((1, 16, 5, 256, 256)).to(device)
    # y = model(x)
    # print(x.shape)
    # print(y.shape)

    model = CNN().to(device)
    x = torch.randn((16, 4, 256, 256), dtype=torch.float64).to(device)
    y = model(x)
    print(x.shape)
    print(y.shape)
