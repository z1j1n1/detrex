export DETECTRON2_DATASETS=/data/jiangning/coco_link
python tools/train_net.py --config-file projects/dino/configs/dino-resnet/dino_r50_5scale_12ep_voc.py --num-gpus 4 dataloader.train.total_batch_size=4  train.output_di ="./output/dino_r50_5scale_12ep_crowdhuman_bs4" model.num_classes=1

