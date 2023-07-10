fp_model_urls = open("ckpt_url_addon.txt","w")
fp = open("config_model_pair_addon.txt","r")
commands = open("commands_addon.sh","w")

commands.write("export DETECTRON2_DATASETS=/data/jiangning/coco_link/\n")

for line in fp.readlines():
    config_url = line.split(' ')[0]
    model_url = line.split(' ')[-1]
    fp_model_urls.write(model_url)

    config_dir = '/'.join([*config_url.split('/')[7:-1],config_url.split('/')[-1].strip()])
    model_name = '.'.join(model_url.split('/')[-1].split('.')[:-1]).strip()
    # model_name = model_name.replace('-','_')

    # One command
    # python tools/train_net.py --config-file {} --num-gpus 4  --eval-only train.init_checkpoint="./checkpoints/{}"

    commands.write(f"python tools/train_net.py --config-file {config_dir} --num-gpus 4  --eval-only train.init_checkpoint='./checkpoints/{model_name}.pth' dataloader.evaluator.output_dir='./eval_boxes_addon/{model_name}' \n")