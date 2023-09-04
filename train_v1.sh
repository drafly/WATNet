pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
pip install pytorch_msssim -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
pip install pytorch_wavelets -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
#pip install tb-nightly -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
#pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn

#pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tsinghua.ustc.edu.cn
#bithaub "/data/xiuxiu/comb/unnormalized3000/train/256/"
#30  /data/dataset/cpf/1122/paired/unnormalized3000

python train_v1.py --images_path "/data/xiuxiu/LOL1/our485" \
       --images_val_path       "/data/xiuxiu/LOL1/eval15/" \
       --snapshots_folder   "/output/STAR-DCE" \
       --out_folder          "/output/"  \
       --snapshots_folder     /output/STAR-DCE \
       --num_epochs 300 \
       --mir_n_feat  16 \
       --mir_chan_factor  2 \
       --train_batch_size  8 \
       --max_lr      1e-3 \
       --max_weight_decay  0.005 \
       --model STAR-MAX-H \
       --snapshot_iter 100 \
       --num_workers 1 \
       --save_img True  \
##       --pretrain_dir        "./pre/psnr_best.pth" \
#       --load_pretrain  True 