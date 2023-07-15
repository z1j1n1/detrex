export DETECTRON2_DATASETS=/data/jiangning/coco_link/voc_0712_detrex/VOCdevkit
python tools/train_net.py --config-file projects/dino/configs/dino-vitdet/dino_vitdet_base_4scale_12ep_voc.py --num-gpus 8 model.num_classes=20 dataloader.train.total_batch_size=8 train.output_dir="./output_train/dino_vitdet_base_12ep_voc"
export DETECTRON2_DATASETS=/data/jiangning/coco_link/
python tools/train_net.py --config-file projects/dino/configs/dino-vitdet/dino_vitdet_base_4scale_12ep_crowdhuman.py --num-gpus 8 model.num_classes=1 dataloader.train.total_batch_size=8 train.output_dir="./output_train/dino_vitdet_base_12ep_crowdhuman"
export DETECTRON2_DATASETS=/data/jiangning/coco_link/voc_0712_detrex/VOCdevkit
python tools/train_net.py --config-file projects/dino/configs/dino-resnet/dino_r50_5scale_12ep_voc.py --num-gpus 8 model.num_classes=20 dataloader.train.total_batch_size=8 train.output_dir="./output_train/dino_r50_5scale_12ep_voc"
export DETECTRON2_DATASETS=/data/jiangning/coco_link
python tools/train_net.py --config-file projects/dino/configs/dino-resnet/dino_r50_5scale_12ep_crowdhuman.py --num-gpus 8  model.num_classes=1 dataloader.train.total_batch_size=8 train.output_dir="./output_train/dino_r50_5scale_12ep_crowdhuman"
export DETECTRON2_DATASETS=/data/jiangning/coco_link/voc_0712_detrex/VOCdevkit
python tools/train_net.py --config-file projects/dab_deformable_detr/configs/dab_deformable_detr_r50_two_stage_50ep_voc.py --num-gpus 8 model.num_classes=20 model.criterion.num_classes=20 dataloader.train.total_batch_size=8 train.output_dir="./output_train/dab_deformable_detr_r50_two_stage_50ep_voc"