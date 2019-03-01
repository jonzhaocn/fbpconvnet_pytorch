from model import FBPCONVNet
from utils import load_checkpoint, load_data, cmap_convert, rsnr
import os
import torch
import math
import torchvision
import argparse
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def eval(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fbp_conv_net = FBPCONVNet().to(device)

    if not (os.path.exists(config.checkpoint_dir) and len(os.listdir(config.checkpoint_dir)) > 0):
        print('load checkpoint unsuccessfully')
        return

    fbp_conv_net, _, _ = load_checkpoint(fbp_conv_net, optimizer=None, checkpoint_dir=config.checkpoint_dir)
    fbp_conv_net.eval()

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
            noisy_image_path = os.path.join(config.eval_result_dir, '%d-noisy.jpg' % (i * config.batch_size + j + 1))
            pred_image_path = os.path.join(config.eval_result_dir, '%d-pred.jpg' % (i*config.batch_size+j+1))
            orig_image_path = os.path.join(config.eval_result_dir, '%d-orig.jpg' % (i*config.batch_size+j+1))

            if config.cmap_convert:
                noisy_image = cmap_convert(noisy_batch[j].squeeze())
                noisy_image.save(noisy_image_path)
                print('save image:', noisy_image_path)

                pred_image = cmap_convert(y_pred[j].squeeze())
                pred_image.save(pred_image_path)
                print('save image:', pred_image_path)

                orig_image = cmap_convert(orig_batch[j].squeeze())
                orig_image.save(orig_image_path)
                print('save image:', orig_image_path)

                SNR = rsnr(np.array(pred_image), np.array(orig_image))
                print('%d-pred.jpg SNR:%f' % (i * config.batch_size + j + 1, SNR))

            else:
                torchvision.utils.save_image(noisy_batch[j].squeeze(), noisy_image_path)
                print('save image:', noisy_image_path)
                torchvision.utils.save_image(y_pred[j].squeeze(), pred_image_path)
                print('save image:', pred_image_path)
                torchvision.utils.save_image(orig_batch[j].squeeze(), orig_image_path)
                print('save image:', orig_image_path)

                pred_image = y_pred[j].clone().detach().cpu().squeeze()
                orig_image = orig_batch[j].clone().detach().cpu().squeeze()
                SNR = rsnr(np.array(pred_image), np.array(orig_image))
                print('%d-pred.jpg SNR:%f' % (i * config.batch_size + j + 1, SNR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--data_path', type=str, default='./preproc_x20_ellipse_fullfbp.mat')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_result_dir', type=str, default='./eval_results')
    parser.add_argument('--cmap_convert', type=bool, default=True)
    config = parser.parse_args()
    print(config)
    eval(config)
