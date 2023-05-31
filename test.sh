#cd STAR-DCE
#pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple/
#pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple/
#pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple/
python test.py \
        --images_path  "/data/dataset/wqq/LOL/" \
        --list_path "/data/dataset/wqq/LOL/test.txt" \
        --snapshots_folder  ./output/ \
        --val_batch_size 1 \
        --pretrain_dir ./pre/psnr_best.pth \
        --model  STAR-MAX-H \
        --mir_chan_factor 2