import os
#23
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import time
from dataloaders.dataloader_FiveK import EnhanceDataset_LOL,EnhanceDataset_UPE
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import random
import models
from options.train_options import TrainOptions
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# os.environ["WANDB_SILENT"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * torch.sum(y_pred * y_true, axes)
    denominator = torch.sum((y_pred) ** 2 + (y_true) ** 2, axes)

    return 1 - torch.mean(numerator / (denominator + epsilon))


def eval(config,epoch):
    print("*************************eval here:*******************************************")
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    DCE_net = models.build_model(config).cuda()  #star-dce-base

    is_yuv = False

    DCE_net.apply(weights_init)

    timestr = datetime.now().strftime('%Y%m%d-%H%M%S')

    if config.parallel:
        print('Using DataParallel')
        print(config.parallel)
        DCE_net = nn.DataParallel(DCE_net)
    if config.pretrain_dir is not None:
        print('Loading {}'.format(config.pretrain_dir))
        state_dict = torch.load(config.pretrain_dir)
#        new_state_dict = OrderedDict()
#        for k, v in state_dict.items():
#            name = 'module.'+ k # remove `module.`
#            new_state_dict[name] = v
        
    DCE_net.load_state_dict(state_dict, strict=True)
    eval_dataset = EnhanceDataset_UPE(config.images_val_path, config.image_h, config.image_w,config.list_path,
                                      mode="test")
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=1,
        pin_memory=False, drop_last=False)

    L_L1 = nn.L1Loss()

    DCE_net.eval()
    psnr_sum = 0
    ssim_sum = 0
    count = 0
    img_idx = 0
    if config.save_img:
        try:
            os.makedirs(config.snapshots_folder + '/results')
        except:
            pass
    batch_time_sum = 0
    
    #warm up
    input_data =torch.randn(1, 3, 400, 600).cuda()
    #input_resize = torch.randn(1, 1, 256, 256).cuda()
    for i in range(100):
        out_image, o_r = DCE_net(input_data)

        flops = FlopCountAnalysis(DCE_net, input_data)
        print("GFLOPs: ", flops.total() / 1e9)

    
    for iteration, img_lowlight in enumerate(eval_loader):
        img_input, img_ref ,input_name,ref_name = img_lowlight
        img_input = img_input.cuda()
        img_ref = img_ref.cuda()
        
        # matte = matte.cuda()
        start = time.time()
        torch.cuda.synchronize()

        
        enhanced_image, x_r = DCE_net(img_input)

        torch.cuda.synchronize()
        batch_time = time.time() - start
        batch_time_sum += batch_time

        # img_output = tensor2img(enhanced_image[0], bit=8)
        # img_gt = tensor2img(img_ref[0], bit=8)

        if True:  # config.save_img:
            # torchvision.utils.save_image(enhanced_image[0],
            #                              '{}/results/{}_out.png'.format(config.snapshots_folder, img_idx))
            # torchvision.utils.save_image(img_ref[0],
            #                              '{}/results/{}_gt.png'.format(config.snapshots_folder, img_idx))
            img_idx += 1
            print("hello")
    print("===============================================")
    print("batch_time is:",batch_time_sum/len(eval_loader))

        # for i in range(img_input.shape[0]):
        #     img_output = tensor2img(enhanced_image[i].detach(), bit=8)
        #     img_gt = tensor2img(img_ref[i], bit=8)
        #     psnr = calculate_psnr(img_output, img_gt)
        #     ssim = calculate_ssim(img_output, img_gt)
        #     count += 1
        #     psnr_sum += psnr
        #     ssim_sum += ssim

            # if not os.path.isdir(config.snapshots_folder+'/results/'):
            #     os.makedirs(config.snapshots_folder+'/results/')
            # if  True: #config.save_img:
            #     torchvision.utils.save_image(enhanced_image[i],
            #                                  '{}/results/{}_out.png'.format(config.snapshots_folder, img_idx))
            #     torchvision.utils.save_image(img_ref[i],
            #                                  '{}/results/{}_gt.png'.format(config.snapshots_folder, img_idx))
            #     img_idx += 1

#         if iteration == 0:
#             A = x_r
#             n, _, h, w = A.shape
#             A = A.sub(A.view(n, _, -1).min(dim=-1)[0].view(n, _, 1, 1)).div(
#                 A.view(n, _, -1).max(dim=-1)[0].view(n, _, 1, 1) - A.view(n, _, -1).min(dim=-1)[0].view(n,
#                                                                                                         _,
#                                                                                                         1,
#                                                                                                         1))
#             writer.add_image('input_enhanced_ref_residual',
#                              torch.cat([img_input[0], enhanced_image[0], img_ref[0],
#                                         torch.abs(
#                                             enhanced_image[0] - img_ref[0])] + [torch.stack(
#                                  (A[0, i], A[0, i], A[0, i]), 0) for i in range(A.shape[1])],
#                                        2
#                                        ), epoch)


#         if (iteration % config.display_iter) == 0:
#             istep = epoch * len(eval_loader) + iteration
#             print( "Batch time: ", batch_time, "PSNR: ", psnr_sum / count, " SSIM: ", ssim_sum / count,
#                   "Batch Time AVG: ", batch_time_sum / (iteration + 1))
#             writer.add_scalar('loss', loss, istep)

#             # import pdb; pdb.set_trace()

#         if ((iteration) % config.snapshot_iter) == 0:
#             torch.save(DCE_net.state_dict(), config.snapshots_folder +
#                        "/Epoch_latest_blendparam_mean_corrected461_test.pth")
#     print("Batch time: ", batch_time, "PSNR: ", psnr_sum / count, " SSIM: ", ssim_sum / count, "Batch Time AVG: ", batch_time_sum / (iteration + 1))
    #wandb.log({"test_PSNR":psnr_sum / count,"test_SSIM":  ssim_sum / count , "Batch Time AVG: ": batch_time_sum / (iteration + 1)})




if __name__ == "__main__":
    option =TrainOptions().parse()
    models.MODEL_REGISTRY[option.model]

    if not os.path.exists(option.snapshots_folder):
        os.makedirs(option.snapshots_folder)
    with torch.no_grad():
        eval(option,0)
