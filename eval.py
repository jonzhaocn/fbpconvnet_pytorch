from model import FBPCONVNet
from utils import load_checkpoint, load_data
import os
import torch
import math
import torchvision
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def eval(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fbp_conv_net = FBPCONVNet().to(device)

    if not (os.path.exists(config.checkpoint_dir) and len(os.listdir(config.checkpoint_dir)) > 0):
        print('load checkpoint unsuccessfully')
        return

    fbp_conv_net, _, _ = load_checkpoint(fbp_conv_net, optimizer=None, checkpoint_dir=config.checkpoint_dir)

    print('load testing data')
    noisy, orig = load_data(config.data_path, device, mode='eval')

    if not os.path.exists(config.eval_result_dir):
        os.mkdir(config.eval_result_dir)

    for i in range(math.ceil(noisy.shape[0]/config.batch_size)):
        i_start = i
        i_end = min(i+config.batch_size, noisy.shape[0])
        noisy_batch = noisy[i_start:i_end]
        orig_batch = orig[i_start:i_end]

        y_pred = fbp_conv_net(noisy_batch)
        for j in range(y_pred.shape[0]):
            image_path = os.path.join(config.eval_result_dir, '%d-pred.jpg' % (i*config.batch_size+j+1))
            torchvision.utils.save_image(y_pred[j].squeeze(), image_path)
            print('save image:', image_path)

            image_path = os.path.join(config.eval_result_dir, '%d-orig.jpg' % (i*config.batch_size+j+1))
            torchvision.utils.save_image(orig_batch[j].squeeze(), image_path)
            print('save image:', image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--data_path', type=str, default='./preproc_x20_ellipse_fullfbp.mat')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_result_dir', type=str, default='./eval_results')
    config = parser.parse_args()
    print(config)
    eval(config)
