export DETECTRON2_DATASETS=/data/jiangning/coco_link/voc_0712_detrex/VOCdevkit
CUDA VISIBLE_DEVICES=2,3 python tools/train_net.py --config-file projects/dino/configs/dino-vitdet/dino_vitdet_base_4scale_12ep_voc_test2012.py --num-gpus 2 --eval-only train.init_checkpoint=/data/jiangning/detrex/output_train/dino_vitdet_base_12ep_voc_bs4/model_final.pth
