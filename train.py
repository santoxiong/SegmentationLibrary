import segmentation_models_pytorch as smp
import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):
    # 数据加载
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 数据划分
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])  # 神器
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # 优化算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if cl > 1 else 'max', patience=2)

    # 损失函数
    if cl > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # 一轮
    for epoch in range(epochs):
        net.train()
        # 一批
        for batch in train_loader:
            imgs = batch['image']
            true_masks = batch['mask']

            # 保证 img 和 mask 维度相等
            assert imgs.shape[1] == ic, \
                f'Network has been defined with {ic} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            # 输入 img 和 mask 到设备
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if cl == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            # 计算 true 和 pre 损失并写入日志
            masks_pred = net(imgs)
            loss = criterion(masks_pred, true_masks)

            # 梯度下降
            optimizer.zero_grad()  # 梯度历史信息清零
            loss.backward()  # 反向传播计算梯度
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()  # 根据计算的梯度进行一次优化(权重)

            val_score = eval_net(net, val_loader, device)
            scheduler.step(val_score)

        # 保存权重文件
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')


# 用户自定义参数
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    ic = 3
    cl = 1
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = smp.Unet(encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                   encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                   in_channels=ic,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                   classes=cl)

    net.to(device=device)

    train_net(net=net,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              device=device,
              img_scale=args.scale,
              val_percent=args.val / 100)
