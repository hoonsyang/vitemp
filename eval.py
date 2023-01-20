# public
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# custom
from DCL_finegrained import model
from utee import selector
from utee.Normalize import Normalize
from loader_checkpoint import *

# logging
import logging, json
with open("logging_config.json", "rt") as file:
    config = json.load(file)
logging.config.dictConfig(config)
logger = logging.getLogger()


parser = argparse.ArgumentParser(description="Transferable Perturbation via Frequency Manipulation")
parser.add_argument("--epochs", type=int, default=8, help="Model checkpoint epoch number")
parser.add_argument("--eps", type=int, default=10, help="Perturbation budget (0~255)")
parser.add_argument("--model_type", type=str, default="vgg16", help="Victim model: vgg16, vgg19, res152, dense169")
parser.add_argument("--RN", type=lambda x: (str(x).lower() == "true"), default=False, help="If true, activating the Random Normalization module in training phase")
parser.add_argument("--DA", type=lambda x: (str(x).lower() == "true"), default=False, help="If true, activating the Domain-agnostic Attention module in training phase")
parser.add_argument("--FA", type=lambda x: (str(x).lower() == "true"), default=True, help="If true, activating the Frequency Augmentation module in training phase")
parser.add_argument("--iter_ckpt", type=lambda x: (str(x).lower() == "true"), default=False, help="If true, Model checkpoint with iteration number")
parser.add_argument("--iter", type=int, default=10000, help="Model checkpoint iteration number")
args = parser.parse_args(args=[])
logger.info(args)

# Choose the domain sets for evaluating the cross-domain transferability
domain_list = ["cifar10", "cifar100", "stl10", "svhn", "dcl_dub", "dcl_car", "dcl_air", "imagenet", "imagenet_incv3"]
domain_selected = domain_list[:4]
# domain_selected = domain_list[-2:-1] # imagenet
logger.info(domain_selected)


for domain in domain_selected:
    # logger.info("="*30, "{}".format(domain), "="*30)  
    logger.info("{}".format(domain))  
    
    # Load the victim model (imagenet)
    if domain[:3] == "dcl": # CUB-200-2011, Stanford Cars, FGVC Aircraft
        batch_size = 6 
        if domain == "dcl_cub":
            numcls = 200
        elif domain == "dcl_car":
            numcls = 196    
        elif domain == "dcl_air":
            numcls = 100
    elif domain == "imagenet_incv3":
        batch_size = 16 
    elif domain == "imagenet": 
        batch_size = 32 
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        model_vgg16 = nn.Sequential(Normalize(mean,std),torchvision.models.vgg16(pretrained=True)).cuda().eval()
        model_vgg19 = nn.Sequential(Normalize(mean,std),torchvision.models.vgg19(pretrained=True)).cuda().eval()
        model_res50 = nn.Sequential(Normalize(mean,std),torchvision.models.resnet50(pretrained=True)).cuda().eval()
        model_res152 = nn.Sequential(Normalize(mean,std),torchvision.models.resnet152(pretrained=True)).cuda().eval()
        model_dense121 = nn.Sequential(Normalize(mean,std),torchvision.models.densenet121(pretrained=True)).cuda().eval()
        model_dense169 = nn.Sequential(Normalize(mean,std),torchvision.models.densenet169(pretrained=True)).cuda().eval()
    else: # CIFAR-10, CIFAR-100, STL-10, SVHN
        batch_size = 128

    # Load the validation dataset & victim model (others)
    if domain == "imagenet":
        ds_fetcher, is_imagenet = selector.select(domain)
    elif domain[:3] == "dcl": # CUB-200-2011, Stanford Cars, FGVC Aircraft
        model_res50, model_senet, model_seres101, ds_fetcher, is_imagenet = selector.select(domain)
        acc_res50, clean_res50, acc_senet, clean_senet, acc_seres101, clean_seres101 = 0,0,0,0,0,0
    else: # CIFAR-10, CIFAR-100, STL-10, SVHN
        model_raw, ds_fetcher, is_imagenet = selector.select(domain)

    if domain[-5:] == "incv3":
        ds_val = ds_fetcher(batch_size=batch_size, input_size=299, train=False, val=True)
        data_length = len(ds_fetcher(batch_size=1, train=False, val=True))
    else:
        ds_val = ds_fetcher(batch_size=batch_size, train=False, val=True)
        data_length = len(ds_fetcher(batch_size=1, train=False, val=True))
    logger.info("Validation data length: {}".format(data_length))
    
    # Load the generative model (attacker)
    netG = load_gan(args, domain).cuda().eval()
    
    # Initialize the classification accuracy (clean & attack)
    clean_vgg16, clean_vgg19, clean_res50, clean_res152, clean_dense121, clean_dense169 = 0,0,0,0,0,0
    acc_vgg16, acc_vgg19, acc_res50, acc_res152, acc_dense121, acc_dense169 = 0,0,0,0,0,0
    clean, accuracy = 0, 0
    
    # Evaluation loop
    for i, data_val in tqdm(enumerate(ds_val)):
        img, label = data_val
        img =  Variable(torch.FloatTensor(img)).cuda()
        label = Variable(torch.from_numpy(np.array(label)).long().cuda())
        adv = netG(img)

        # projection
        adv = torch.min(torch.max(adv, img - args.eps/255.0), img + args.eps/255.0)
        adv = torch.clamp(adv, 0.0, 1.0)


        with torch.no_grad():
            if domain == "imagenet":
                clean_vgg16 += torch.sum(torch.argmax(model_vgg16(img), dim = 1) == label.cuda())
                acc_vgg16 += torch.sum(torch.argmax(model_vgg16(adv), dim = 1) == label.cuda()) 

                clean_vgg19 += torch.sum(torch.argmax(model_vgg19(img), dim = 1) == label.cuda())
                acc_vgg19 += torch.sum(torch.argmax(model_vgg19(adv), dim = 1) == label.cuda())

                clean_res50 += torch.sum(torch.argmax(model_res50(img), dim = 1) == label.cuda())
                acc_res50 += torch.sum(torch.argmax(model_res50(adv), dim = 1) == label.cuda())
                
                clean_res152 += torch.sum(torch.argmax(model_res152(img), dim = 1) == label.cuda())
                acc_res152 += torch.sum(torch.argmax(model_res152(adv), dim = 1) == label.cuda())

                clean_dense121 += torch.sum(torch.argmax(model_dense121(img), dim = 1) == label.cuda())
                acc_dense121 += torch.sum(torch.argmax(model_dense121(adv), dim = 1) == label.cuda())

                clean_dense169 += torch.sum(torch.argmax(model_dense169(img), dim = 1) == label.cuda())
                acc_dense169 += torch.sum(torch.argmax(model_dense169(adv), dim = 1) == label.cuda())  

            elif domain[:3] != "dcl":
                clean += torch.sum(torch.argmax(model_raw(img), dim = 1) == label.cuda())
                accuracy += torch.sum(torch.argmax(model_raw(adv), dim = 1) == label.cuda())
            else:
                outputs = model_res50(adv)
                outputs_clean = model_res50(img)
                outputs_pred = outputs[0] + outputs[1][:,0:numcls] + outputs[1][:,numcls:2*numcls]
                outputs_pred_clean = outputs_clean[0] + outputs_clean[1][:,0:numcls] + outputs_clean[1][:,numcls:2*numcls]
                acc_res50 += torch.sum(torch.argmax(outputs_pred, dim = 1) == label.cuda())
                clean_res50 += torch.sum(torch.argmax(outputs_pred_clean, dim = 1) == label.cuda())

                outputs2 = model_senet(adv)
                outputs_clean2 = model_senet(img)
                outputs_pred2 = outputs2[0] + outputs2[1][:,0:numcls] + outputs2[1][:,numcls:2*numcls]
                outputs_pred_clean2 = outputs_clean2[0] + outputs_clean2[1][:,0:numcls] + outputs_clean2[1][:,numcls:2*numcls]
                acc_senet += torch.sum(torch.argmax(outputs_pred2, dim = 1) == label.cuda())
                clean_senet += torch.sum(torch.argmax(outputs_pred_clean2, dim = 1) == label.cuda())

                outputs3 = model_seres101(adv)
                outputs_clean3 = model_seres101(img)
                outputs_pred3 = outputs3[0] + outputs3[1][:,0:numcls] + outputs3[1][:,numcls:2*numcls]
                outputs_pred_clean3 = outputs_clean3[0] + outputs_clean3[1][:,0:numcls] + outputs_clean3[1][:,numcls:2*numcls]
                acc_seres101 += torch.sum(torch.argmax(outputs_pred3, dim = 1) == label.cuda())
                clean_seres101 += torch.sum(torch.argmax(outputs_pred_clean3, dim = 1) == label.cuda())
    

    # if domain == "imagenet":
    #     logger.info("----------------vgg16----------------")
    #     logger.info(clean_vgg16 / data_length)
    #     logger.info(acc_vgg16 / data_length)
    #     logger.info("----------------vgg19----------------")
    #     logger.info(clean_vgg19 / data_length)
    #     logger.info(acc_vgg19 / data_length)
    #     logger.info("----------------res50----------------")
    #     logger.info(clean_res50 / data_length)
    #     logger.info(acc_res50 / data_length)      
    #     logger.info("----------------res152----------------")
    #     logger.info(clean_res152 / data_length)
    #     logger.info(acc_res152 / data_length)
    #     logger.info("----------------dense121----------------")
    #     logger.info(clean_dense121 / data_length)
    #     logger.info(acc_dense121 / data_length)
    #     logger.info("----------------dense169----------------")
    #     logger.info(clean_dense169 / data_length)
    #     logger.info(acc_dense169 / data_length)
        
    # elif domain[:3] == "dcl": # CUB-200-2011, Stanford Cars, FGVC Aircraft
    #     logger.info("----------------backbone:res50----------------")
    #     logger.info(clean_res50 / data_length)
    #     logger.info(acc_res50 / data_length)
    #     logger.info("----------------backbone:se-net----------------")
    #     logger.info(clean_senet / data_length)
    #     logger.info(acc_senet / data_length)
    #     logger.info("----------------backbone:se-res101----------------")
    #     logger.info(clean_seres101 / data_length)
    #     logger.info(acc_seres101 / data_length)

    # else: # CIFAR-10, CIFAR-100, STL-10, SVHN
    #     logger.info(clean / data_length)
    #     logger.info(accuracy / data_length)


    if domain == "imagenet":
        logger.info("vgg16")
        logger.info(clean_vgg16 / data_length)
        logger.info(acc_vgg16 / data_length)
        logger.info("vgg19")
        logger.info(clean_vgg19 / data_length)
        logger.info(acc_vgg19 / data_length)
        logger.info("res50")
        logger.info(clean_res50 / data_length)
        logger.info(acc_res50 / data_length)      
        logger.info("res152")
        logger.info(clean_res152 / data_length)
        logger.info(acc_res152 / data_length)
        logger.info("dense121")
        logger.info(clean_dense121 / data_length)
        logger.info(acc_dense121 / data_length)
        logger.info("dense169")
        logger.info(clean_dense169 / data_length)
        logger.info(acc_dense169 / data_length)
        
    elif domain[:3] == "dcl": # CUB-200-2011, Stanford Cars, FGVC Aircraft
        logger.info("backbone:res50")
        logger.info(clean_res50 / data_length)
        logger.info(acc_res50 / data_length)
        logger.info("backbone:se-net")
        logger.info(clean_senet / data_length)
        logger.info(acc_senet / data_length)
        logger.info("backbone:se-res101")
        logger.info(clean_seres101 / data_length)
        logger.info(acc_seres101 / data_length)

    else: # CIFAR-10, CIFAR-100, STL-10, SVHN
        logger.info(clean / data_length)
        logger.info(accuracy / data_length)