export DETECTRON2_DATASETS=/data/jiangning/coco_link/
python tools/train_net.py --config-file projects/dino/configs/dino-vitdet/dino_vitdet_base_4scale_12ep_voc.py --num-gpus 8
