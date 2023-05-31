import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    
    def initialize(self):
        self.parser.add_argument('--mir_path', type=str, default="Enhancement_MIRNet_v2_Lol.yml") #Enhancement_MIRNet_v2_Lol.yml

        self.parser.add_argument('--snapshots_folder', type=str, default="./output/STAR-DCE")
        self.parser.add_argument('--out_folder', type=str, default="./output/")
        self.parser.add_argument('--dataset', type=str,     default='fivek')
        self.parser.add_argument('--pretrain_dir', type=str, default="./output/STAR-DCE/Epoch_latest.pth")
        self.parser.add_argument('--model', type=str, default='STAR-MAX-H')
        self.parser.add_argument('--star_lr',     type=float, default=0.005)
        self.parser.add_argument('--star_lr_type', type=str, default='cos')

        self.parser.add_argument('--star_weight_decay', type=float, default=0.0001)
        self.parser.add_argument('--star_grad_clip_norm', type=float, default=1)

        self.parser.add_argument('--num_epochs', type=int, default=300)
        self.parser.add_argument('--train_batch_size', type=int, default=8)
        self.parser.add_argument('--val_batch_size', type=int, default=1)
        self.parser.add_argument('--warmup_epochs', type=int, default=10)

        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--display_iter', type=int, default=10)
        self.parser.add_argument('--snapshot_iter', type=int, default=10)

        self.parser.add_argument('--list_path', type=str, default="")

        self.parser.add_argument('--load_pretrain', type=bool, default=False)
        self.parser.add_argument('--save_img', type=bool, default=False)
        self.parser.add_argument('--parallel', type=bool, default=False)
        self.parser.add_argument('--zerodce', type=bool, default=False)
        self.parser.add_argument('--tv_loss', type=bool, default=False)
        self.parser.add_argument('--color_loss', type=bool, default=False)
        self.parser.add_argument('--l2_loss', type=bool, default=False)
        self.parser.add_argument('--cos_loss', type=bool, default=False)
        self.parser.add_argument('--mul_loss', type=bool, default=False)
        self.parser.add_argument('--no_l1', type=bool, default=False)

        self.parser.add_argument('--image_h', type=int, default=400)
        self.parser.add_argument('--image_w', type=int, default=600)
        self.parser.add_argument('--image_ds', type=int, default=256)


        self.parser.add_argument('--gpu_ids', type=str, default='1, 0')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--depth', type=int, default=8, help='# of input/output image depth')


        #NAF Params
        ###network
        self.parser.add_argument('--width', type=int, default=32)
        self.parser.add_argument('--enc_blk_nums', type=list, default=[2, 2, 2, 2])
        self.parser.add_argument('--middle_blk_num', type=int, default=2)
        self.parser.add_argument('--dec_blk_nums', type=list, default=[2, 2, 2, 2])

        # MIRNetv2 Params
        #Mirnet的使用方法
        ###network
        self.parser.add_argument('--mir_inp_channels', type=int, default=3)
        self.parser.add_argument('--mir_out_channels', type=int, default=3)
        self.parser.add_argument('--mir_n_feat', type=int, default = 16)
        self.parser.add_argument('--mir_chan_factor', type=float, default= 1.5)

        self.parser.add_argument('--mir_n_RRG', type=int, default=1)
        self.parser.add_argument('--mir_n_MRB', type=int, default=1)
        self.parser.add_argument('--mir_height', type=int, default=4)
        self.parser.add_argument('--mir_width', type=int, default=2)
        self.parser.add_argument('--mir_scale', type=int, default=1)
        
        
        self.parser.add_argument('--mir_lr', type=float, default=2e-4)
        self.parser.add_argument('--mir_weight_decay', type=float, default=0.)
        self.parser.add_argument('--mir_betas', type=list, default=[0.9, 0.999])


        ###optimizer
        self.parser.add_argument('--naf_lr', type=float, default=1e-3)
        self.parser.add_argument('--naf_weight_decay', type=float, default=0.)
        self.parser.add_argument('--naf_betas', type=list, default=[0.9, 0.9])

        ### max optimizer
        self.parser.add_argument('--max_lr', type=float, default=1e-3)
        self.parser.add_argument('--max_weight_decay', type=float, default=0)
        self.parser.add_argument('--max_betas', type=list, default=[0.9, 0.999])
        self.parser.add_argument('--max_lr_min', type=float, default=0)

        #Star Params
        self.parser.add_argument('--number_f', type =int, default=32)
        self.parser.add_argument('--star_heads', type=int, default=8)
        self.parser.add_argument('--star_depth', type=int, default=8)
        self.parser.add_argument('--star_dropout', type=float, default=0.0)
        self.parser.add_argument('--star_patch_num', type=int, default=32)

        #IAT Params
        # self.parser.add_argument('--no_vgg_instance', action='store_true', help='vgg instance normalization')
        # self.parser.add_argument('--vgg_model_dir', type=str, default='/data/kepler/vgg16/')
        

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        # self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        #if len(self.opt.gpu_ids) > 0:
        #    torch.cuda.set_device(self.opt.gpu_ids[0])#self.opt.gpu_ids[0]

        args = vars(self.opt)

        #设置输出路径
        if not os.path.exists(self.opt.snapshots_folder):
            os.makedirs(self.opt.snapshots_folder)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt

        


        

        
