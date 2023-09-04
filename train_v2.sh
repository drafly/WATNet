pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
pip install pytorch_msssim -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn

python train_v2.py --images_path "/data/xiuxiu/LOLv2/Real_captured/Train_patch" \
       --images_val_path       "/data/xiuxiu/LOLv2/Real_captured/Test/" \
       --snapshots_folder   "/output/STAR-DCE" \
       --out_folder          "/output/"  \
       --snapshots_folder     /output/STAR-DCE \
       --num_epochs 300 \
       --mir_n_feat  16 \
       --mir_chan_factor  2 \
       --train_batch_size  8 \
       --max_lr      5e-4 \
       --max_weight_decay  0.005 \
       --model STAR-MAX-H \
       --snapshot_iter 100 \
       --num_workers 1 \
       --save_img True  \
##       --pretrain_dir        "./pre/psnr_best.pth" \
#       --load_pretrain  True 