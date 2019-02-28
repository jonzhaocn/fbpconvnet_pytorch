import torch
import torch.nn as nn


class FBPCONVNet(nn.Module):
    def __init__(self):
        super(FBPCONVNet, self).__init__()
        # create network model
        self.block_1_1 = None
        self.block_2_1 = None
        self.block_3_1 = None
        self.block_4_1 = None
        self.block_5 = None
        self.block_4_2 = None
        self.block_3_2 = None
        self.block_2_2 = None
        self.block_1_2 = None
        self.create_model()

    def forward(self, input):
        block_1_1_output = self.block_1_1(input)
        block_2_1_output = self.block_2_1(block_1_1_output)
        block_3_1_output = self.block_3_1(block_2_1_output)
        block_4_1_output = self.block_4_1(block_3_1_output)
        block_5_output = self.block_5(block_4_1_output)
        result = self.block_4_2(torch.cat((block_4_1_output, block_5_output), dim=1))
        result = self.block_3_2(torch.cat((block_3_1_output, result), dim=1))
        result = self.block_2_2(torch.cat((block_2_1_output, result), dim=1))
        result = self.block_1_2(torch.cat((block_1_1_output, result), dim=1))
        result = result + input
        return result

    def create_model(self):
        kernel_size = 3
        padding = kernel_size // 2

        # block_1_1
        block_1_1 = []
        block_1_1.extend(self.add_block_conv(in_channels=1, out_channels=64, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_1.extend(self.add_block_conv(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_1.extend(self.add_block_conv(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_1_1 = nn.Sequential(*block_1_1)

        # block_2_1
        block_2_1 = [nn.MaxPool2d(kernel_size=2)]
        block_2_1.extend(self.add_block_conv(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_1.extend(self.add_block_conv(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_2_1 = nn.Sequential(*block_2_1)

        # block_3_1
        block_3_1 = [nn.MaxPool2d(kernel_size=2)]
        block_3_1.extend(self.add_block_conv(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_1.extend(self.add_block_conv(in_channels=256, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_3_1 = nn.Sequential(*block_3_1)

        # block_4_1
        block_4_1 = [nn.MaxPool2d(kernel_size=2)]
        block_4_1.extend(self.add_block_conv(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_1.extend(self.add_block_conv(in_channels=512, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_4_1 = nn.Sequential(*block_4_1)

        # block_5
        block_5 = [nn.MaxPool2d(kernel_size=2)]
        block_5.extend(self.add_block_conv(in_channels=512, out_channels=1024, kernel_size=kernel_size, stride=1,
                                           padding=padding, batchOn=True, ReluOn=True))
        block_5.extend(self.add_block_conv(in_channels=1024, out_channels=1024, kernel_size=kernel_size, stride=1,
                                           padding=padding, batchOn=True, ReluOn=True))
        block_5.extend(self.add_block_conv_transpose(in_channels=1024, out_channels=512, kernel_size=kernel_size, stride=2,
                                                     padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_5 = nn.Sequential(*block_5)

        # block_4_2
        block_4_2 = []
        block_4_2.extend(self.add_block_conv(in_channels=1024, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_2.extend(self.add_block_conv(in_channels=512, out_channels=512, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_4_2.extend(
            self.add_block_conv_transpose(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_4_2 = nn.Sequential(*block_4_2)

        # block_3_2
        block_3_2 = []
        block_3_2.extend(self.add_block_conv(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_2.extend(self.add_block_conv(in_channels=256, out_channels=256, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_3_2.extend(
            self.add_block_conv_transpose(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_3_2 = nn.Sequential(*block_3_2)

        # block_2_2
        block_2_2 = []
        block_2_2.extend(self.add_block_conv(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_2.extend(self.add_block_conv(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_2_2.extend(
            self.add_block_conv_transpose(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=2,
                                          padding=padding, output_padding=1, batchOn=True, ReluOn=True))
        self.block_2_2 = nn.Sequential(*block_2_2)

        # block_1_2
        block_1_2 = []
        block_1_2.extend(self.add_block_conv(in_channels=128, out_channels=64, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_2.extend(self.add_block_conv(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        block_1_2.extend(self.add_block_conv(in_channels=64, out_channels=1, kernel_size=kernel_size, stride=1,
                                             padding=padding, batchOn=True, ReluOn=True))
        self.block_1_2 = nn.Sequential(*block_1_2)

    @staticmethod
    def add_block_conv(in_channels, out_channels, kernel_size, stride, padding, batchOn, ReluOn):
        seq = []
        seq.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding))
        if batchOn:
            seq.append(nn.BatchNorm2d(num_features=out_channels))
        if ReluOn:
            seq.append(nn.ReLU())
        return seq

    @staticmethod
    def add_block_conv_transpose(in_channels, out_channels, kernel_size, stride, padding, output_padding, batchOn, ReluOn):
        seq = []
        seq.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding))
        if batchOn:
            seq.append(nn.BatchNorm2d(num_features=out_channels))
        if ReluOn:
            seq.append(nn.ReLU())
        return seq
