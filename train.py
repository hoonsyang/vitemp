# Public
import os
import sys
import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Custom
from model_layer import Vgg16_all_layer, Vgg19_all_layer, Res152_all_layer, Dense169_all_layer
from generator import GeneratorResnet
from dct import *
from utils import *
from loader_checkpoint import *

# Argument Parser
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/home/vilab/yhm/dataset/imagenet/train', help='the path for imagenet training data') 
    parser.add_argument('--batch_size', type=int, default=16, help='Number of trainig samples/batch')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
    parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
    parser.add_argument('--model_type', type=str, default='vgg16',
                        help='Model against GAN is trained: vgg16, vgg19 res152, dense169')
    parser.add_argument('--RN', type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help='If true, activating the Random Normalization module in training phase')
    parser.add_argument('--DA', type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help='If true, activating the Domain-agnostic Attention module in training phase')
    
    parser.add_argument('--FA', type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help='If true, activating the Frequency-space Augmentation module in training phase')
    parser.add_argument("--rho", type=float, default=0.5, help="Tuning factor")
    parser.add_argument("--sigma", type=float, default=16.0, help="Std of random noise")
    parser.add_argument('--iter_ckpt', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If true, Model checkpoint with iteration number')
    parser.add_argument('--iter', type=int, default=10000, help='Save Model checkpoint iteration number')
    parser.add_argument('--load_pretrained_G', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('LOGS', 'logs'), help="output folder")
    parser.add_argument('--exp', type=str, default='BIA+FA', help='exp name')
    parser.add_argument('--random_seed', type=int, default=1, help='random seed (default: 1)')
    
    return parser


def main():
    global logger
    args = get_argparser().parse_args()
    rank=0
    mkdir(args.output_dir)
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    # Setup logger
    logger = setup_logger("BIA+FA", args.output_dir, rank)
    logger.info("--------------------------------------------------------------")
    logger.info("Experiments: %s" % args.exp)
    logger.info("Device: %s" % device)
    logger.info("Training arguments %s" % args)
    
    # Setup random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    ####################
    # Victim Model
    ####################
    if args.model_type == 'vgg16':
        model = Vgg16_all_layer.Vgg16()
        layer_idx = 16  # Maxpooling.3
    elif args.model_type == 'vgg19':
        model = Vgg19_all_layer.Vgg19()
        layer_idx = 18  # Maxpooling.3
    elif args.model_type == 'res152':
        model = Res152_all_layer.Resnet152()
        layer_idx = 5   # Conv3_8
    elif args.model_type == 'dense169':
        model = Dense169_all_layer.Dense169()
        layer_idx = 6  # Denseblock.2
    else:
        raise Exception('Please check the model_type')

    model = model.to(device)
    model.eval()

    ####################
    # Generator Model, Optimizer
    ####################
    if args.RN and args.DA:
        save_checkpoint_suffix = 'BIA+RN+DA'
    elif args.RN:
        save_checkpoint_suffix = 'BIA+RN'
    elif args.DA:
        save_checkpoint_suffix = 'BIA+DA'
    elif args.FA:
        save_checkpoint_suffix = 'BIA+FA'
    else:
        save_checkpoint_suffix = 'BIA'  


    netG = GeneratorResnet()
    if args.load_pretrained_G:
        pretrained_checkpoint_suffix = 'BIA'
        pretrained_epochs = 0
        pretrained_checkpoint_path = '../saved_models_GT/{}/netG_{}_{}.pth'.format(args.model_type, pretrained_checkpoint_suffix, pretrained_epochs)
        pretrained_checkpoint = torch.load(pretrained_checkpoint_path, map_location=torch.device('cpu'))
        netG.load_state_dict(pretrained_checkpoint)
        logger.info("Load the pretrained checkpoint from %s" % pretrained_checkpoint_path)
    else:
        logger.info("Training from scratch!")
    
    # netG = nn.DataParallel(netG, device_ids=[0,1,2,3])
    netG = netG.to(device)
        
    netG_params = sum(param.numel() for param in netG.parameters() if param.requires_grad)
    logger.info('Number of generator (learnable) parameters: {}'.format(netG_params))
    
    optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    
    # Data, Transform
    scale_size = 256
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    train_dir = args.train_dir
    train_set = datasets.ImageFolder(train_dir, data_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    train_size = len(train_set)
    logger.info('Training data size: {}'.format(train_size))


    def default_normalize(t):
        t[:, 0, :, :] = (t[:, 0, :, :] - 0.485) / 0.229
        t[:, 1, :, :] = (t[:, 1, :, :] - 0.456) / 0.224
        t[:, 2, :, :] = (t[:, 2, :, :] - 0.406) / 0.225

        return t

    def normalize(t, mean, std):
        t[:, 0, :, :] = (t[:, 0, :, :] - mean) / std
        t[:, 1, :, :] = (t[:, 1, :, :] - mean) / std
        t[:, 2, :, :] = (t[:, 2, :, :] - mean) / std

        return t


    # Training
    save_checkpoint_dir = args.output_dir
    
    # save_checkpoint_dir = 'saved_models/{}'.format(args.model_type)
    # if not os.path.exists(save_checkpoint_dir):
    #     os.makedirs(save_checkpoint_dir)

    for epoch in range(args.epochs):
        running_loss = 0
        for i, (img, _) in enumerate(train_loader):
            img = img.to(device)
            netG.train()
            optimG.zero_grad()
            
            if args.FA and i%2==1:
                gauss = (torch.randn(img.size()[0], 3, img_size, img_size) * (args.sigma / 255)).to(device)
                mask = (torch.rand_like(img) * 2 * args.rho + 1 - args.rho).to(device)
                img_dct = dct_2d(img + gauss).to(device)
                img_idct = idct_2d(img_dct * mask)
                img_idct = V(img_idct, requires_grad=True)
                img = img_idct
            else:
                pass
            
            # adversarial translation        
            adv = netG(img)
            adv = torch.min(torch.max(adv, img - args.eps/255.0), img + args.eps/255.0)
            adv = torch.clamp(adv, 0.0, 1.0)

            # Saving adversarial examples
            flag = False
            if i <= 1000:
                if i % 100 == 0:
                    flag = True
            else:
                if i % 2000 == 0:
                    flag = True
            if flag:
                plt.subplot(121)
                plt.imshow(img[0,...].permute(1,2,0).detach().cpu())
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(adv[0,...].permute(1,2,0).detach().cpu())
                plt.axis('off')
                save_path = args.output_dir
                # save_path = 'saved_models/output/{}'.format(args.model_type)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(os.path.join(save_path, '{}.png'.format(i)), bbox_inches='tight')

            if args.RN:
                mean = np.random.normal(0.50, 0.08)
                std = np.random.normal(0.75, 0.08)
                adv_out_slice = model(normalize(adv.clone(), mean, std))[layer_idx]
                img_out_slice = model(normalize(img.clone(), mean, std))[layer_idx]
            else:
                adv_out_slice = model(default_normalize(adv.clone()))[layer_idx]
                img_out_slice = model(default_normalize(img.clone()))[layer_idx]

            
            if args.DA:
                attention = abs(torch.mean(img_out_slice, dim=1, keepdim=True)).detach()
            else:
                attention = torch.ones(adv_out_slice.shape).cuda()


            loss = torch.cosine_similarity((adv_out_slice*attention).reshape(adv_out_slice.shape[0], -1), 
                                (img_out_slice*attention).reshape(img_out_slice.shape[0], -1)).mean()
            loss.backward()
            optimG.step()


            # Every 100 iteration
            if i % 100 == 0:
                logger.info('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 100))
                running_loss = 0
            running_loss += abs(loss.item())

            # Every 1 epoch 
            if args.iter_ckpt:
                if i % args.iter == 0 and i > 0:
                    save_path = os.path.join(save_checkpoint_dir, 'netG_{}_{}_{}.pth'.format(save_checkpoint_suffix, epoch, i))
                
                    if isinstance(netG, nn.DataParallel):
                        torch.save(netG.module.state_dict(), save_path)
                    else:
                        torch.save(netG.state_dict(), save_path)
                    
            else:
                if i % 80000 == 0 and i > 0: # 1epoch=80000batch                        
                    save_path = os.path.join(save_checkpoint_dir, 'netG_{}_{}.pth'.format(save_checkpoint_suffix, epoch))
        
                    if isinstance(netG, nn.DataParallel):
                        torch.save(netG.module.state_dict(), save_path)
                    else:
                        torch.save(netG.state_dict(), save_path)


if __name__ == '__main__':
    main()