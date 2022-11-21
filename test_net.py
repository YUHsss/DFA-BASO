import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import time
from PIL import Image

from data import test_dataset
from models.My.Mynet import *

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args()

dataset_path = r'E:\data\EORSSD\test/'

model = Net()

model.load_state_dict(torch.load('./model/my_net/Net_epoch45_EORSSD.pth'))

model.cuda()
model.eval()

test_datasets = ['EORSSD']
# test_datasets = ['ORSSD']

for dataset in test_datasets:
    save_path = './results/Mynet/' + dataset + '/'
    rgb_path = r"E:\c_run\MyNet-main\results\Mynet/RGB_EORSSD/"
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + '/image/'
    # print(dataset)
    gt_root = dataset_path + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        time_start = time.time()
        output = model(image)
        time_end = time.time()
        time_sum = time_sum + (time_end - time_start)
        res = F.upsample(output[0], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        print('save img to: ', save_path + name)
        img_rgb = Image.fromarray(res * 255).convert('RGB')
        cv2.imwrite(save_path + name, res * 255)
        img_rgb.save(rgb_path + name)
        if i == test_loader.size - 1:
            print('Running time {:.5f}'.format(time_sum / test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size / time_sum))
