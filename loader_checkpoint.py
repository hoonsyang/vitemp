import torch
import torchvision
from generator import GeneratorResnet
import pandas as pd
import torch.nn as nn

# Load a particular generator
def load_gan(args, domain): 
    if domain[-5:] == 'incv3': 
        netG = GeneratorResnet(inception=True)
    else:
        netG = GeneratorResnet()
    
    netG = nn.DataParallel(netG, device_ids=[0,1,2,3]) # multi-GPU
        
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
    
    
    print('Substitute Model: {} \t RN: {} \t DA: {} \t FA: {} \t Saving instance: {}'.format(args.model_type,
                                                                                  args.RN,
                                                                                  args.DA,
                                                                                  args.FA,
                                                                                  args.epochs))
    
    if args.iter_ckpt:
        if isinstance(netG, nn.DataParallel):
            netG.module.load_state_dict(torch.load('saved_models/{}/netG_{}_{}_{}.pth'.format(args.model_type,
                                                                                save_checkpoint_suffix,
                                                                                args.epochs,
                                                                                args.iter)))
        else:
            netG.load_state_dict(torch.load('saved_models/{}/netG_{}_{}_{}.pth'.format(args.model_type,
                                                                                save_checkpoint_suffix,
                                                                                args.epochs,
                                                                                args.iter)))           
    
    else:
        if isinstance(netG, nn.DataParallel):
            netG.module.load_state_dict(torch.load('saved_models/{}/netG_{}_{}.pth'.format(args.model_type,
                                                                                save_checkpoint_suffix,
                                                                                args.epochs)))
        else:
            netG.load_state_dict(torch.load('saved_models/{}/netG_{}_{}.pth'.format(args.model_type,
                                                                                save_checkpoint_suffix,
                                                                                args.epochs)))                                                                                                           
   
    return netG
