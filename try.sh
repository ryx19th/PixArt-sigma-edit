


python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 \
    train_scripts/train.py \
    configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms.py \
    --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
    --work-dir output/your_first_pixart-exp \
    --debug




python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
    train_scripts/train.py \
    configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms_anyedit.py \
    --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
    --work-dir output/anyedit



python scripts/inference.py \
    --image_size 512 \
    --model_path output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
    --txt_file asset/samples.txt


python scripts/inference.py \
    --image_size 512 \
    --model_path output/your_first_pixart-exp/checkpoints/epoch_10_step_441.pth \
    --txt_file asset/samples.txt


python scripts/inference.py \
    --image_size 512 \
    --model_path output/anyedit/checkpoints/epoch_100_step_2929.pth \
    --txt_file asset/samples.txt




python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
    train_scripts/train.py \
    configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms_anyedit_edit.py \
    --load-from output/anyedit_edit/checkpoints/epoch_100_step_2929.pth \
    --work-dir output/anyedit_edit


python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
    train_scripts/train.py \
    configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms_anyedit_edit.py \
    --resume-from output/anyedit_edit_all/checkpoints/epoch_1_step_2600.pth \
    --work-dir output/anyedit_edit_all


python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
    train_scripts/train.py \
    configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms_anyedit_edit.py \
    --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
    --work-dir output/anyedit_edit_all_x


python -m torch.distributed.launch --nproc_per_node=8 --master_port=12345 \
    train_scripts/train.py \
    configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms_anyedit_edit.py \
    --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
    --work-dir output/anyedit_edit_all_s