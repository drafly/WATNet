from models import register_model
from models.transformer import *
from models.VIT import NCB,NTB
from models.naf_models.NAFNet_arch import NAFNet
from models.mir_models.archs.mirnet_v2_arch import MIRNet_v2
from models.max_models.maxvit import MaxViTTransformerBlock,grid_partition,grid_reverse



@register_model('STAR-DCE-Ori')
class enhance_net_litr(nn.Module):

    def __init__(self, number_f=32, depth=1, heads=8, dropout=0, patch_num=32):
        super(enhance_net_litr, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(1, number_f // 2, 2, 3, 1, 1, bias=True)#输入的通道数改为1

        self.patch_num = patch_num
        self.transformer = Transformer(number_f // 2, depth, heads, number_f // 2, dropout) #transformer输入的只有16
        self.ncb_block   = NCB(16,16,1,0,0,8,3)
        self.ntb_block   = NTB(16,16,0.1,1,1,2,2,mix_block_ratio=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, number_f // 2, 32, 32))

        self.enc_conv = nn.Conv2d(number_f // 2, number_f // 2, 3, 1, 1, bias=True)    #CNN分支输入的只有16
        self.out_conv = nn.Conv2d(number_f, 8, 3, 1, 1, bias=True) #输出的通道数改为8
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.LinearSqueeze = nn.Linear(64,16)
        print("--------------------------------------------------------------------------------------")


    @staticmethod
    def add_args(parser):
        parser.add_argument("--number-f", type=int, default=32, help="number of features")
        parser.add_argument("--depth", type=int, default=1, help="depth of transformer block")
        parser.add_argument("--heads", type=int, default=8, help="heads of transformer block")
        parser.add_argument("--dropout", type=float, default= 0., help="dropout ratio of transformer block")
        parser.add_argument("--patch-num", type=int, default=32, help="number of patches")

    @classmethod
    def build_model(cls, args):
        return cls(number_f=args.number_f, depth=args.depth, heads=args.heads, dropout=args.dropout,
                   patch_num=args.patch_num)

    def forward(self, x_in, img_in=None):
        torch.autograd.set_detect_anomaly(True)
        patch_size = 8
        
        x_in_single_1 = x_in  #x_in_single_1 [8,1,256,256]
        img_in_1 = img_in
        img_in = x_in_single_1 if img_in is None else img_in_1
        n, c, h, w = x_in.shape

        patches = x_in_single_1.unfold(2,patch_size,patch_size).unfold(3,patch_size,patch_size)
        patches =patches.squeeze(dim =1)
        patches = patches.contiguous().view(x_in_single_1.shape[0]*1*patches.shape[1]*patches.shape[2],-1)
        patches = self.LinearSqueeze(patches)

        #将64个像素转换为通道数，与通道1交换，并将数目为1的维度压缩
        x1 = patches.view(x_in_single_1.shape[0],32,32,16).permute(0,3,1,2)

        #cnn branch ,the aim is detail refine
        ## transformer branch ,the aim is to make input's shape is B (hw) C ,then send it to trans
        x_conv = self.ncb_block(x1)
        x_out = self.ntb_block(x1)


        x_r = torch.cat([x_out, x_conv], dim=1)
        x_r = self.out_conv(x_r)
        x_r = F.upsample_bilinear(x_r, (256, 256)).tanh()
        x_r_resize = F.interpolate(x_r, img_in.shape[2:], mode='bilinear')
        
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r_resize, 1, dim=1)

        x = img_in.to(x_r_resize.device)
        x = x + r1 * (torch.pow(x, 2) - x  )
        x = x + r2 * (torch.pow(x, 2) - x  )
        x = x + r3 * (torch.pow(x, 2) - x )
        enhanced_image_1 = x + r4 * (torch.pow(x, 2) - x )
        x = enhanced_image_1 + r5 * (torch.pow(enhanced_image_1, 2) - enhanced_image_1 )
        x = x + r6 * (torch.pow(x, 2) - x )
        x = x + r7 * (torch.pow(x, 2) - x )
        enhanced_image = x + r8 * (torch.pow(x, 2) - x  )
        
        enhanced_image = torch.clamp(enhanced_image, 0, 1)   
        # enhanced_image = torch.cat((enhanced_image,enhanced_image,enhanced_image),dim = 1)           
        return enhanced_image, x_r_resize


@register_model('STAR-DCE-NAF')
class enhance_net_litr(nn.Module):

    def __init__(self, opt):
        super(enhance_net_litr, self).__init__()

        self.channels = opt.input_nc
        self.relu = nn.ReLU(inplace=False)

        self.e_conv1 = nn.Conv2d(1, opt.number_f // 2, 2, 3, 1, 1, bias=True)  # 输入的通道数改为1


        self.transformer = Transformer(opt.number_f // 2, opt.star_depth, opt.star_heads, opt.number_f // 2, opt.star_dropout)  # transformer输入的只有16

        # self.naf_block = NAFNet(img_channel=self.channels, width=opt.width, middle_blk_num=opt.middle_blk_num,
        #               enc_blk_nums=opt.enc_blk_nums, dec_blk_nums=opt.dec_blk_nums)
        self.naf_block = NAFNet(img_channel= opt.input_nc , width=opt.width, middle_blk_num=opt.middle_blk_num,
                                enc_blk_nums=opt.enc_blk_nums, dec_blk_nums=opt.dec_blk_nums)

        self.ntb_block = NTB(16, opt.input_nc, 0.1, 1, 1, 2,  opt.input_nc, mix_block_ratio=1)

        self.pos_embedding = nn.Parameter(torch.randn(1, opt.number_f // 2, 32, 32))
        self.enc_conv = nn.Conv2d(opt.number_f // 2, opt.number_f // 2, 3, 1, 1, bias=True)  # CNN分支输入的只有16
        self.out_conv = nn.Conv2d(opt.input_nc*2, 8, 3, 1, 1, bias=True)  # 输出的通道数改为8
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.LinearSqueeze = nn.Linear(64, 16)
        self.GlobalLiner = nn.Linear(1024*opt.input_nc,opt.input_nc)


        self.MutualBlock = MGB(1)
        print("--------------------------------------------------------------------------------------")


    @classmethod
    def build_model(cls, args):
        return cls(args)

    def forward(self, x_in, img_in=None):
        torch.autograd.set_detect_anomaly(True)
        patch_size = 8

        x_in_single_1 = x_in  # x_in_single_1 [8,1,256,256]
        img_in_1 = img_in
        img_in = x_in_single_1 if img_in is None else img_in_1

        patches = x_in_single_1.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.squeeze(dim=1)
        patches = patches.contiguous().view(x_in_single_1.shape[0] * 1 * patches.shape[1] * patches.shape[2], -1)
        patches = self.LinearSqueeze(patches)

        # 将64个像素转换为通道数，与通道1交换，并将数目为1的维度压缩
        x1 = patches.view(x_in_single_1.shape[0], 32, 32, 16).permute(0, 3, 1, 2)

        # cnn branch ,the aim is detail refine
        ## transformer branch ,the aim is to make input's shape is B (hw) C ,then send it to trans
        x_conv = self.relu(self.naf_block(img_in_1))

        y_out = self.ntb_block(x1)
        y_reshape = rearrange(y_out, 'b c h w -> b (h w c)')

        y_mutual = self.GlobalLiner(y_reshape).unsqueeze(2)
        x_mutual = nn.functional.adaptive_avg_pool2d(x_conv, (1,1)).squeeze(2)


        mutual_matrix =self.MutualBlock(x_mutual,y_mutual)

        x_ =  torch.matmul(rearrange(x_conv, "b c h w -> b (h w ) c"),mutual_matrix)
        n, c, h, w = x_in.shape
        x_ =  rearrange(x_, "b (h w ) c-> b c h w ",h=h)

        #### NO resize, Directly Linear
        #x_r = F.upsample_bilinear(x_r, (256, 256)).tanh()
        #x_r_resize = F.interpolate(x_r, img_in.shape[2:], mode='bilinear')

        ####NO cat
        #x_r_cat =torch.cat([x_conv,x_r_resize],dim=1)
        y_r = F.upsample_bilinear(y_out, (256, 256)).tanh()
        x_r = torch.cat([x_, y_r], dim=1)

        x_end = self.out_conv(x_r)

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_end, 1, dim=1)

        x = x_conv.to(x_end.device)
        x = x + r1 * (torch.pow(x, 2) - x  )
        x = x + r2 * (torch.pow(x, 2) - x  )
        x = x + r3 * (torch.pow(x, 2) - x )
        enhanced_image_1 = x + r4 * (torch.pow(x, 2) - x )
        x = enhanced_image_1 + r5 * (torch.pow(enhanced_image_1, 2) - enhanced_image_1 )
        x = x + r6 * (torch.pow(x, 2) - x )
        x = x + r7 * (torch.pow(x, 2) - x )
        enhanced_image = x + r8 * (torch.pow(x, 2) - x  )
        enhanced_image = torch.clamp(enhanced_image, 0, 1)

        # enhanced_image = torch.cat((enhanced_image,enhanced_image,enhanced_image),dim = 1)
        return enhanced_image, x_end





@register_model('STAR-DCE-MIR')
class enhance_net_litr(nn.Module):

    def __init__(self, opt):
        super(enhance_net_litr, self).__init__()

        self.channels = opt.input_nc
        self.relu = nn.ReLU(inplace=False)

        self.e_conv1 = nn.Conv2d(3, opt.number_f // 2, 2, 3, 1, 1, bias=True)  # 输入的通道数改为1


        self.transformer = Transformer(opt.number_f // 2, opt.star_depth, opt.star_heads, opt.number_f // 2, opt.star_dropout)  # transformer输入的只有16

        # self.naf_block = NAFNet(img_channel=self.channels, width=opt.width, middle_blk_num=opt.middle_blk_num,
        #               enc_blk_nums=opt.enc_blk_nums, dec_blk_nums=opt.dec_blk_nums)
        # self.naf_block = NAFNet(img_channel= opt.input_nc , width=opt.width, middle_blk_num=opt.middle_blk_num,
        #                         enc_blk_nums=opt.enc_blk_nums, dec_blk_nums=opt.dec_blk_nums)

        self.mir_block = MIRNet_v2(opt.mir_inp_channels,opt.mir_out_channels,opt.mir_n_feat,opt.mir_chan_factor,opt.mir_n_RRG,
                                   opt.mir_n_MRB,opt.mir_height,opt.mir_width,opt.mir_scale)

        self.ntb_block = NTB(16*self.channels, opt.input_nc, 0.1, 1, 1, 2,  opt.input_nc, mix_block_ratio=1)

        self.pos_embedding = nn.Parameter(torch.randn(1, opt.number_f // 2, 32, 32))
        self.enc_conv = nn.Conv2d(opt.number_f // 2, opt.number_f // 2, 3, 1, 1, bias=True)  # CNN分支输入的只有16
        self.out_conv = nn.Conv2d(opt.input_nc*2, 24, 3, 1, 1, bias=True)  # 输出的通道数改为8
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.LinearSqueeze = nn.Linear(64, 16)
        self.GlobalLiner = nn.Linear(1024*opt.input_nc,opt.input_nc)


        self.MutualBlock = MGB(1)
        print("--------------------------------------------------------------------------------------")


    @classmethod
    def build_model(cls, args):
        return cls(args)

    def forward(self, x_in, img_in=None):
        torch.autograd.set_detect_anomaly(True)
        patch_size = 8

        x_in_single_1 = x_in  # x_in_single_1 [8,1,256,256]
        img_in_1 = img_in
        img_in = x_in_single_1 if img_in is None else img_in_1
        
        #单色图部分
#         patches = x_in_single_1.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#         patches = patches.squeeze(dim=1)
#         patches = patches.contiguous().view(x_in_single_1.shape[0] * 1 * patches.shape[1] * patches.shape[2], -1)
#         patches = self.LinearSqueeze(patches)
#         x1 = patches.view(x_in_single_1.shape[0], 32, 32, 16).permute(0, 3, 1, 2)# 将64个像素转换为通道数，与通道1交换，并将数目为1的维度压
        
        patches = x_in_single_1.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(x_in_single_1.shape[0] * patches.shape[3] * patches.shape[1] * patches.shape[2], -1)
        patches = self.LinearSqueeze(patches)
        
        x1 = patches.view(-1, 32, 32, 16*self.channels).permute(0, 3, 1, 2)

        # cnn branch ,the aim is detail refine
        ## transformer branch ,the aim is to make input's shape is B (hw) C ,then send it to trans
        x_conv = self.mir_block(img_in_1)
        y_out = self.ntb_block(x1)
        
        y_reshape = rearrange(y_out, 'b c h w -> b (c h w )')
    
        y_mutual = self.GlobalLiner(y_reshape).unsqueeze(2) 
        x_mutual = nn.functional.adaptive_avg_pool2d(x_conv, (1,1)).squeeze(3)


        mutual_matrix = self.relu(self.MutualBlock(x_mutual,y_mutual).unsqueeze(3).expand(-1,-1,400,600))

        x_ =  (x_conv * mutual_matrix).tanh()
        n, c, h, w = img_in.shape


        
        y_r = F.upsample_bilinear(y_out, (256, 256)).tanh()
        y_r = F.interpolate(y_r, img_in.shape[2:], mode='bilinear')
        
        x_r = torch.cat([x_, y_r], dim=1)


        x_end = self.relu(self.out_conv(x_r))

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_end, 3, dim=1)

        x = x_conv.to(x_end.device)
        x = x + r1 * (torch.pow(x, 2) - x  )
        x = x + r2 * (torch.pow(x, 2) - x  )
        x = x + r3 * (torch.pow(x, 2) - x )
        enhanced_image_1 = x + r4 * (torch.pow(x, 2) - x )
        x = enhanced_image_1 + r5 * (torch.pow(enhanced_image_1, 2) - enhanced_image_1 )
        x = x + r6 * (torch.pow(x, 2) - x )
        x = x + r7 * (torch.pow(x, 2) - x )
        enhanced_image = x + r8 * (torch.pow(x, 2) - x  )
        enhanced_image = torch.clamp(enhanced_image, 0, 1)

        # enhanced_image = torch.cat((enhanced_image,enhanced_image,enhanced_image),dim = 1)
        return enhanced_image, x_end


@register_model('STAR-DCE-MAX')
class enhance_net_litr(nn.Module):

    def __init__(self, opt):
        super(enhance_net_litr, self).__init__()

        self.channels = opt.input_nc
        self.relu = nn.ReLU(inplace=False)

        self.e_conv1 = nn.Conv2d(3, opt.number_f // 2, 2, 3, 1, 1, bias=True)  # 输入的通道数改为1

        self.transformer = Transformer(opt.number_f // 2, opt.star_depth, opt.star_heads, opt.number_f // 2,
                                       opt.star_dropout)  # transformer输入的只有16

        self.max_block = MaxViTTransformerBlock(in_channels= opt.input_nc ,partition_function= grid_partition,reverse_function=grid_reverse,
            num_heads=1, #32
            grid_window_size=(8,8), #(7,7)
            attn_drop=0.,
            drop=0.,
            drop_path=0.,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer= nn.LayerNorm )

        self.max_block_2 = MaxViTTransformerBlock(in_channels=opt.input_nc, partition_function=grid_partition,
                                                reverse_function=grid_reverse,
                                                num_heads=1,  # 32
                                                grid_window_size=(8, 8),  # (7,7)
                                                attn_drop=0.,
                                                drop=0.,
                                                drop_path=0.,
                                                mlp_ratio=4.,
                                                act_layer=nn.GELU,
                                                norm_layer=nn.LayerNorm)

        self.mir_block = MIRNet_v2(opt.mir_inp_channels, opt.mir_out_channels, opt.mir_n_feat, opt.mir_chan_factor,
                                   opt.mir_n_RRG,
                                   opt.mir_n_MRB, opt.mir_height, opt.mir_width, opt.mir_scale)

        self.ntb_block = NTB(16 * self.channels, opt.input_nc, 0.1, 1, 1, 2, opt.input_nc, mix_block_ratio=1)

        self.pos_embedding = nn.Parameter(torch.randn(1, opt.number_f // 2, 32, 32))
        self.enc_conv = nn.Conv2d(opt.number_f // 2, opt.number_f // 2, 3, 1, 1, bias=True)  # CNN分支输入的只有16
        self.out_conv = nn.Conv2d(opt.input_nc * 2, 24, 3, 1, 1, bias=True)  # 输出的通道数改为8
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.LinearSqueeze = nn.Linear(64, 16)

        self.MutualBlock = MGB(1)
        print("--------------------------------------------------------------------------------------")

    @classmethod
    def build_model(cls, args):
        return cls(args)

    def forward(self, x_in, img_in=None):
        torch.autograd.set_detect_anomaly(True)
        patch_size = 8

        x_in_single_1 = x_in  # x_in_single_1 [8,1,256,256]
        img_in_1 = img_in
        img_in = x_in_single_1 if img_in is None else img_in_1

        # 单色图部分
        #         patches = x_in_single_1.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        #         patches = patches.squeeze(dim=1)
        #         patches = patches.contiguous().view(x_in_single_1.shape[0] * 1 * patches.shape[1] * patches.shape[2], -1)
        #         patches = self.LinearSqueeze(patches)
        #         x1 = patches.view(x_in_single_1.shape[0], 32, 32, 16).permute(0, 3, 1, 2)# 将64个像素转换为通道数，与通道1交换，并将数目为1的维度压

        patches = x_in_single_1.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(
            x_in_single_1.shape[0] * patches.shape[3] * patches.shape[1] * patches.shape[2], -1)
        patches = self.LinearSqueeze(patches)

        x1 = patches.view(-1, 32, 32, 16 * self.channels).permute(0, 3, 1, 2)

        # cnn branch ,the aim is detail refine
        ## transformer branch ,the aim is to make input's shape is B (hw) C ,then send it to trans
        x_conv = self.mir_block(img_in_1)
        y_out = self.ntb_block(x1)
        ###################################

        y_max_out =self.max_block(img_in_1)
        y_max_out_2 =self.max_block_2(y_max_out) + y_max_out



        ##################

        #y_reshape = rearrange(y_max_out_2, 'b c h w -> b (c h w )')
        y_mutual = nn.functional.adaptive_avg_pool2d(y_max_out_2, (1, 1)).squeeze(3)
        x_mutual = nn.functional.adaptive_avg_pool2d(x_conv, (1, 1)).squeeze(3)

        mutual_matrix = self.relu(self.MutualBlock(x_mutual, y_mutual).unsqueeze(3).expand(-1, -1, 400, 600))
        mutual_matrix_y = self.relu(self.MutualBlock(y_mutual, x_mutual).unsqueeze(3).expand(-1, -1, 400, 600))

        x_ = (x_conv * mutual_matrix).tanh()
        y_ = (y_max_out_2*mutual_matrix_y).tanh()

        x_r = torch.cat([x_, y_], dim=1)
        x_end = self.relu(self.out_conv(x_r))

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_end, 3, dim=1)

        x = x_conv.to(x_end.device)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhanced_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhanced_image_1 + r5 * (torch.pow(enhanced_image_1, 2) - enhanced_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhanced_image = x + r8 * (torch.pow(x, 2) - x)
        enhanced_image = torch.clamp(enhanced_image, 0, 1)

        # enhanced_image = torch.cat((enhanced_image,enhanced_image,enhanced_image),dim = 1)
        return enhanced_image, x_end


@register_model('STAR-MAX-H')
class enhance_net_litr(nn.Module):

    def __init__(self, opt):
        super(enhance_net_litr, self).__init__()
        self.relu = nn.ReLU(inplace=False)

        self.relu = nn.Conv2d(3, opt.number_f // 2, 2, 3, 1, 1, bias=True)  # 输入的通道数改为1

        self.relu = MaxViTTransformerBlock(in_channels= opt.input_nc ,partition_function= grid_partition,reverse_function=grid_reverse,
            num_heads=1, #32
            grid_window_size=(8,8), #(7,7)
            attn_drop=0.,
            drop=0.,
            drop_path=0.,
            mlp_ratio=4.,
            act_layer=nn.GELU,
            norm_layer= nn.LayerNorm )

        self.mir_block = MIRNet_v2(opt.mir_inp_channels, opt.mir_out_channels, opt.mir_n_feat, opt.mir_chan_factor,
                                   opt.mir_n_RRG,
                                   opt.mir_n_MRB, opt.mir_height, opt.mir_width, opt.mir_scale)
        
        self.relu = nn.ReLU(inplace=False)

        print("--------------------------------------------------------------------------------------")

    @classmethod
    def build_model(cls, args):
        return cls(args)

    def forward(self, x_in):
        torch.autograd.set_detect_anomaly(True)
        patch_size = 8
        # cnn branch ,the aim is detail refine
        ## transformer branch ,the aim is to make input's shape is B (hw) C ,then send it to trans
        x_conv = self.mir_block(x_in)
        x_conv = torch.clamp(x_conv, 0, 1)

        return x_conv,x_conv




#消融实验中的slimmer model
@register_model('STAR-DCE-Half')
class enhance_net_litr_half(nn.Module):

    def __init__(self, number_f=16, depth=1, heads=8, dropout=0, patch_num=32):
        super(enhance_net_litr_half, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, number_f // 2, 2, 3, 1, 1, bias=True)

        self.patch_num = patch_num
        self.transformer = Transformer(number_f // 2, depth, heads, number_f // 2, dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, number_f // 2, 32, 32))

        self.enc_conv = nn.Conv2d(number_f // 2, number_f // 2, 3, 1, 1, bias=True)

        self.out_conv = nn.Conv2d(number_f, 24, 3, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        print("--------------------------------------------------------------------------------------")
        print("STAR-DCE-Half")


    @staticmethod
    def add_args(parser):
        parser.add_argument("--number-f", type=int, default=16, help="number of features")
        parser.add_argument("--depth", type=int, default=1, help="depth of transformer block")
        parser.add_argument("--heads", type=int, default=8, help="heads of transformer block")
        parser.add_argument("--dropout", type=float, default=0., help="dropout ratio of transformer block")
        parser.add_argument("--patch-num", type=int, default=32, help="number of patches")

    @classmethod
    def build_model(cls, args):
        return cls(number_f=args.number_f, depth=args.depth, heads=args.heads, dropout=args.dropout,
                   patch_num=args.patch_num)

    def forward(self, x_in, img_in=None):
        torch.autograd.set_detect_anomaly(True)
        img_in = x_in if img_in is None else img_in

        n, c, h, w = x_in.shape

        x1 = self.e_conv1(x_in)
        x1 = self.relu(x1)

        x_r_d = nn.AdaptiveAvgPool2d((h // 8, w // 8))(x1)
        x_conv = self.enc_conv(x_r_d)
        trans_inp = rearrange(x_r_d, 'b c  h w -> b (  h w)  c ')
        x_out = self.transformer(trans_inp)
        x_r = torch.cat([rearrange(x_out, 'b (h w) c -> b  c  h w', h=self.patch_num), x_conv], dim=1)
        x_r = self.out_conv(x_r)

        x_r = F.upsample_bilinear(x_r, (256, 256)).tanh()

        x_r_resize = F.interpolate(x_r, img_in.shape[2:], mode='bilinear')
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r_resize, 3, dim=1)    
        x = img_in.to(x_r_resize.device)
        x = x.clone() + r1 * (torch.pow(x.clone(), 2) - x.clone())
        x = x.clone() + r2 * (torch.pow(x.clone(), 2) - x.clone())
        x = x.clone() + r3 * (torch.pow(x.clone(), 2) - x.clone())
        enhanced_image_1 = x.clone() + r4 * (torch.pow(x.clone(), 2) - x.clone())
        x = enhanced_image_1.clone() + r5 * (torch.pow(enhanced_image_1.clone(), 2) - enhanced_image_1.clone())
        x = x.clone() + r6 * (torch.pow(x.clone(), 2) - x.clone())
        x = x.clone() + r7 * (torch.pow(x.clone(), 2) - x.clone())
        enhanced_image = x.clone() + r8 * (torch.pow(x.clone(), 2) - x.clone())
        enhanced_image = torch.clamp(enhanced_image.clone(), 0, 1)

        return enhanced_image, x_r_resize

