import torch
import os
import scipy.io as scipy_io
import numpy as np
import re


def load_data(data_path, device, mode):
    data = scipy_io.loadmat(data_path)
    noisy = data['lab_d']
    orig = data['lab_n']
    noisy = np.transpose(noisy, [3, 2, 0, 1])
    orig = np.transpose(orig, [3, 2, 0, 1])

    training_images_count = round(noisy.shape[0]*0.95)
    if mode == 'train':
        noisy = torch.tensor(noisy[0:training_images_count]).float().to(device)
        orig = torch.tensor(orig[0:training_images_count]).float().to(device)
    elif mode == 'eval':
        noisy = torch.tensor(noisy[training_images_count:]).float().to(device)
        orig = torch.tensor(orig[training_images_count:]).float().to(device)
    else:
        raise ValueError('mode should be train or test')
    return noisy, orig


def load_checkpoint(model, optimizer, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        raise ValueError('checkpoint dir does not exist')

    checkpoint_list = os.listdir(checkpoint_dir)
    if len(checkpoint_list) > 0:

        checkpoint_list.sort(key=lambda x: int(re.findall(r"epoch-(\d+).pkl", x)[0]))

        last_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[-1])
        print('load checkpoint: %s' % last_checkpoint_path)

        model_ckpt = torch.load(last_checkpoint_path)
        model.load_state_dict(model_ckpt['state_dict'])

        if optimizer:
            optimizer.load_state_dict(model_ckpt['optimizer'])

        epoch = model_ckpt['epoch']
        return model, optimizer, epoch