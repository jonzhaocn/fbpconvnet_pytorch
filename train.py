from model import FBPCONVNet
import torch
import numpy as np
import math
import argparse
import torchvision
import os
from utils import load_data, load_checkpoint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def data_argument(noisy, orig):

    # flip horizontal
    for i in range(noisy.shape[0]):
        rate = np.random.random()
        if rate > 0.5:
            noisy[i] = noisy[i].flip(2)
            orig[i] = orig[i].flip(2)

    # flip vertical
    for i in range(noisy.shape[0]):
        rate = np.random.random()
        if rate > 0.5:
            noisy[i] = noisy[i].flip(1)
            orig[i] = orig[i].flip(1)
    return noisy, orig


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('load training data')
    noisy, orig = load_data(config.data_path, device=device, mode='train')

    epoch = config.epoch
    batch_size = config.batch_size
    grad_max = config.grad_max
    learning_rate = config.learning_rate

    fbp_conv_net = FBPCONVNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(fbp_conv_net.parameters(), lr=learning_rate[0], momentum=config.momentum)
    epoch_start = 0

    # load check_point
    if os.path.exists(config.checkpoint_dir) and len(os.listdir(config.checkpoint_dir)) > 0:
        fbp_conv_net, optimizer, epoch_start = load_checkpoint(fbp_conv_net, optimizer, config.checkpoint_dir)

    print('start training...')
    for e in range(epoch_start, epoch):

        # each epoch
        for i in range(math.ceil(noisy.shape[0]/batch_size)):
            i_start = i
            i_end = min(i_start+batch_size, noisy.shape[0])
            noisy_batch = noisy[i_start:i_end]
            orig_batch = orig[i_start:i_end]

            # data argument
            noisy_batch, orig_batch = data_argument(noisy_batch, orig_batch)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward Propagation
            y_pred = fbp_conv_net(noisy_batch)

            if (i+1) % config.sample_step == 0:
                if not os.path.exists(config.sample_dir):
                    os.mkdir(config.sample_dir)
                image_path = os.path.join(config.sample_dir, 'epoch-%d-iteration-%d.jpg' % (e + 1, i + 1))
                torchvision.utils.save_image(y_pred.squeeze(), image_path)
                print('save image:', image_path)

            # Compute and print loss
            loss = criterion(y_pred, orig_batch)
            if (i+1) % 100 == 0:
                print('loss (epoch-%d-iteration-%d) : %f' % (e+1, i+1, loss.item()))

            loss.backward()

            # clip gradient
            torch.nn.utils.clip_grad_value_(fbp_conv_net.parameters(), clip_value=grad_max)

            # Update the parameters
            optimizer.step()

        # adjust learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate[min(e+1, len(learning_rate)-1)]

        # save check_point
        if (e+1) % config.checkpoint_save_step == 0 or (e+1) == config.epoch:
            if not os.path.exists(config.checkpoint_dir):
                os.mkdir(config.checkpoint_dir)
            check_point_path = os.path.join(config.checkpoint_dir, 'epoch-%d.pkl' % (e+1))
            torch.save({'epoch': e+1, 'state_dict': fbp_conv_net.state_dict(), 'optimizer': optimizer.state_dict()},
                       check_point_path)
            print('save checkpoint %s', check_point_path)


if __name__ == '__main__':
    epoch = 151
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=epoch)
    parser.add_argument('--learning_rate', type=tuple, default=np.logspace(-2, -3, epoch))
    parser.add_argument('--grad_max', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--data_path', type=str, default='./preproc_x20_ellipse_fullfbp.mat')
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--sample_dir', type=str, default='./samples/')
    parser.add_argument('--checkpoint_save_step', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    config = parser.parse_args()
    main(config)
