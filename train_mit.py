import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim
import os
import time
import torch.nn.functional as F
from dataloaders.dataloader_FiveK import EnhanceDataset_LOL,EnhanceDataset_MIT,EnhanceDataset_UPE
from util import calculate_ssim, calculate_psnr, tensor2img, draw_picture,torchPSNR,torchSSIM
from options.train_options import TrainOptions

#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import models
import tqdm
import losses
import copy
import random
import numpy as np
from warmup_scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt
import numpy
import wandb
plt.switch_backend('agg')

#设置cuda的设备是卡二
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def fix_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True, warn_only=True)




def train(config):
    train_loss_path = os.path.join( config.out_folder,"train_loss.txt")
    loss_png_path   = os.path.join( config.out_folder,"loss.png")
    psnr_path = os.path.join( config.out_folder,"psnr_txt.txt")
    psnr_png  = os.path.join( config.out_folder,"test_psnr.png")  
    ssim_path = os.path.join( config.out_folder,"ssim_txt.txt")  
    ssim_png  = os.path.join( config.out_folder,"test_ssim.png") 
    train_loss =[]
    test_psnr = []
    test_ssim = []
    psnr_sum = 0.0
    ssim_sum = 0.0

    DCE_net = models.build_model(config).cuda()

    ###DCE_net.apply(weights_init)

    L_L1 = nn.L1Loss() if not config.l2_loss else nn.MSELoss()
    L_color = losses.L_color()
    L_psnr = losses.PSNRLoss()


    optimizer_max= torch.optim.AdamW(DCE_net.parameters(), lr=config.max_lr,
                                      betas=config.max_betas, weight_decay=config.max_weight_decay)

    #scheduler_max = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_max, T_max=config.num_epochs* len(train_loader),eta_min = config.max_lr_min)

    warmup_epochs = config.warmup_epochs
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_max, (config.num_epochs - warmup_epochs)*len(train_loader),
                                                            eta_min=float(config.max_lr_min))
    scheduler_max = GradualWarmupScheduler(optimizer_max, multiplier=1, total_epoch=warmup_epochs* len(train_loader),
                                       after_scheduler=scheduler_cosine)

    scheduler_max.step()
    
    
    
    if config.parallel:
        print('Using DataParallel')
        DCE_net = nn.DataParallel(DCE_net)
    
    if config.load_pretrain:
        print('Loading {}'.format(config.pretrain_dir))
        DCE_net.load_state_dict(torch.load(config.pretrain_dir), strict=True)

    
    for epoch in range(config.num_epochs):
        batch_time_sum = 0
        loss_total = 0
        DCE_net.train()
        print("------------------Epoch %d--------------------" % epoch)

        for iteration, img_lowlight in tqdm.tqdm(enumerate(train_loader)):
            optimizer_max.zero_grad()

            img_input, img_ref,input_name,ref_name = img_lowlight
            img_input = img_input.cuda()
            img_ref = img_ref.cuda()

            torch.cuda.synchronize()
            enhanced_image, x_r = DCE_net(img_input)
            torch.cuda.synchronize()

            loss_color = torch.mean(L_color(enhanced_image)) if config.color_loss else torch.zeros([]).cuda()
            loss_l1 = L_L1(enhanced_image, img_ref) if not config.no_l1 else torch.zeros([]).cuda()
            loss_cos = 1 - nn.functional.cosine_similarity(enhanced_image, img_ref,
                                                           dim=1).mean() if config.cos_loss else torch.zeros([]).cuda()
            loss_psnr =L_psnr(enhanced_image,img_ref)


            if config.mul_loss:
                star_loss = loss_l1 * loss_cos
            else:
                star_loss = loss_l1  
            star_loss.backward()

            loss_total = loss_total + star_loss.item()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.star_grad_clip_norm)

            optimizer_max.step()
            scheduler_max.step()


        ### save and log
        train_loss.append(loss_total)
        if loss_total == min(train_loss):
            print('Save best loss {}'.format(loss_total))
            torch.save(DCE_net.state_dict(), config.snapshots_folder + "/Epoch_best.pth")

        torch.save(DCE_net.state_dict(), config.snapshots_folder +"/Epoch_latest.pth")
        with open(train_loss_path,'w') as train_los:
            train_los.write(str(train_loss))
            train_los.close()
        draw_picture(train_loss_path,loss_png_path ,"loss")
        
        
        # eval
        if epoch % 1 == 0:       
            DCE_net.eval()
            psnr_sum = 0.0
            ssim_sum = 0.0
            count = 0
            img_idx = 0
            if not os.path.isdir(config.snapshots_folder + '/results/'):
                os.makedirs(config.snapshots_folder + '/results/')

            for iteration, img_lowlight in enumerate(eval_loader):
                img_input, img_ref,input_name,ref_name = img_lowlight
                img_input = img_input.cuda()
                img_ref = img_ref.cuda()

                with torch.no_grad():
                    enhanced_image, x_r = DCE_net(img_input)
                    
                
                
                for i in range(img_input.shape[0]):
                    pytorch_psnr = torchPSNR(enhanced_image[i],img_ref[i])
                    pytorch_ssim = torchSSIM(enhanced_image,img_ref)
                    count += 1
                    img_idx +=1
                    ssim_sum += pytorch_ssim
                    psnr_sum += pytorch_psnr
                    
                    if  config.save_img : #config.save_img:
                        torchvision.utils.save_image(enhanced_image[i],'{}/results/{}_out.png'.format(config.snapshots_folder, img_idx))
                        torchvision.utils.save_image(img_ref[i],'{}/results/{}_gt.png'.format(config.snapshots_folder, img_idx))

                if (iteration % config.display_iter) == 0:
                    print("iteration:",iteration , "PSNR: ", psnr_sum / count, " SSIM: ", ssim_sum / count)

                if ((iteration) % config.snapshot_iter) == 0:
                    torch.save(DCE_net.state_dict(), config.snapshots_folder +"/Epoch_latest_snap.pth")
            
            epoch_psnr =(psnr_sum/len(eval_loader)).item()
            epoch_ssim =(ssim_sum/len(eval_loader)).item()

            print("iteration:", count, "PSNR: ", epoch_psnr, " SSIM: ", epoch_ssim)

            with open(psnr_path,'w') as psnr_txt:
                test_psnr.append(epoch_psnr)
                psnr_txt.write(str(test_psnr))
                psnr_txt.close()
            draw_picture(psnr_path, psnr_png ,"Tets PSNR")
        
            with open(ssim_path,'w') as ssim_txt:
                test_ssim.append(epoch_ssim)
                ssim_txt.write(str(test_ssim))
                ssim_txt.close()
            draw_picture(ssim_path, ssim_png ,"Test SSIM")

            best_psnr = max(test_psnr)
            best_ssim = max(test_ssim)

            if best_psnr == epoch_psnr:
                torch.save(DCE_net.state_dict(), config.snapshots_folder + "/psnr_best.pth")

            if best_ssim == epoch_ssim:
                torch.save(DCE_net.state_dict(), config.snapshots_folder + "/ssim_best.pth")
        

            with open( os.path.join(config.out_folder + 'best_.txt'),'w') as best_txt:
                best_txt.write("psnr :  "+str(best_psnr)+"   ssim" +str(best_ssim))
                best_txt.close()

        wandb.log({"test_PSNR":psnr_sum / len(eval_loader),"test_SSIM":  ssim_sum / len(eval_loader),"train_loss":loss_total})



if __name__ == "__main__":
    #set_wandb(0)
    
    option =TrainOptions().parse()
    models.MODEL_REGISTRY[option.model]
    config_eval = copy.copy(option)

    ####
    fix_random_seed(2023)

    ###
    train_dataset = EnhanceDataset_UPE(option.images_path, option.image_h, image_size_w=option.image_w,mode="train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=option.train_batch_size, shuffle=False, num_workers=option.num_workers,
        pin_memory=True, drop_last=False)
    print("train_loader load finished ",len(train_dataset) )

    eval_dataset = EnhanceDataset_UPE(
        config_eval.images_val_path,
        config_eval.image_h, config_eval.image_w,mode="test")
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=config_eval.val_batch_size, shuffle=False, num_workers=option.num_workers,
        pin_memory=False, drop_last=False)
    print("eval_loader load finished ",len(eval_dataset) )

    print(option)
    train(option)