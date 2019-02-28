from model import FBPCONVNet
import torch
import numpy as np
import math
import argparse
import scipy.io as sio
import torchvision
import os


def data_argument(images, labels):

    # flip horizontal
    for i in range(images.shape[0]):
        rate = np.random.random()
        if rate > 0.5:
            images[i] = images[i].flip(2)
            labels[i] = labels[i].flip(2)

    # flip vertical
    for i in range(images.shape[0]):
        rate = np.random.random()
        if rate > 0.5:
            images[i] = images[i].flip(1)
            labels[i] = labels[i].flip(1)
    return images, labels


def get_training_data(data_path, device):
    data = sio.loadmat(data_path)
    images = data['lab_d']
    labels = data['lab_n']
    images = np.transpose(images, [3, 2, 0, 1])
    labels = np.transpose(labels, [3, 2, 0, 1])

    training_images_count = round(images.shape[0]*0.95)
    images = torch.tensor(images[0:training_images_count]).float().to(device)
    labels = torch.tensor(labels[0:training_images_count]).float().to(device)
    return images, labels


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images, labels = get_training_data(config.data_path, device=device)

    epoch = config.epoch
    batch_size = config.batch_size
    grad_max = config.grad_max
    learning_rate = config.learning_rate

    fbp_conv_net = FBPCONVNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(fbp_conv_net.parameters(), lr=learning_rate[0], momentum=config.momentum)

    print('start training...')
    for e in range(epoch):

        # each epoch
        for i in range(math.ceil(images.shape[0]/batch_size)):
            i_start = i
            i_end = min(i_start+batch_size, images.shape[0])
            images_batch = images[i_start:i_end]
            labels_batch = labels[i_start:i_end]

            # data argument
            images_batch, labels_batch = data_argument(images_batch, labels_batch)

            # Forward Propagation
            y_pred = fbp_conv_net(images_batch)

            if (i+1) % config.sample_step == 0:
                if not os.path.exists(config.sample_dir):
                    os.mkdir(config.sample_dir)
                image_path = os.path.join(config.sample_dir, 'epoch-%d-iteration-%d.jpg' % (e + 1, i + 1))
                torchvision.utils.save_image(y_pred.squeeze(), image_path)
                print('save image:', image_path)

            # Compute and print loss
            loss = criterion(y_pred, labels_batch)
            if (i+1) % 100 == 0:
                print('loss (epoch-%d-iteration-%d) : %f' % (e+1, i+1, loss.item()))

            # Zero the gradients
            optimizer.zero_grad()
            loss.backward()

            # clip gradient
            torch.nn.utils.clip_grad_value_(fbp_conv_net.parameters(), clip_value=grad_max)

            # Update the parameters
            optimizer.step()

        # adjust learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate[min(e+1, len(learning_rate)-1)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=151)
    parser.add_argument('--learning_rate', type=tuple, default=np.logspace(-2, -3, 151))
    parser.add_argument('--grad_max', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--data_path', type=str, default='./preproc_x20_ellipse_fullfbp.mat')
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--sample_dir', type=str, default='./results/')
    config = parser.parse_args()
    main(config)
