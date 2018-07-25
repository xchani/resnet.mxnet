# mxnet version: https://github.com/huangzehao/incubator-mxnet-bk
mxnet_path = '../incubator-mxnet-bk/python/'
gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]
dataset = "imagenet"
model_prefix = "resnet_hires_50"
depth = 50
model_load_prefix = model_prefix
model_load_epoch = 0
retrain = False
memonger = True
allow_missing = False

high_resolution = True
high_semantics = False
channel_factor = 1
network = "resnet_hires" if high_resolution else "resnet"

# data
data_dir = '/mnt/truenas/scratch/chenxia.han/data/imagenet/'
batch_size = 32
batch_size *= len(gpu_list)
kv_store = 'device'

# optimizer
lr = 0.1
wd = 0.0001
momentum = 0.9
if dataset == "imagenet":
    lr_step = [30, 60, 90]
else:
    lr_step = [120, 160, 240]
lr_factor = 0.1
begin_epoch = model_load_epoch if retrain else 0
num_epoch = 100
frequent = 50

# network config
if dataset == "imagenet":
    num_classes = 1000
    units_dict = {"18": [2, 2, 2, 2],
                  "34": [3, 4, 6, 3],
                  "50": [3, 4, 6, 3],
                  "101": [3, 4, 23, 3],
                  "152": [3, 8, 36, 3]}
    if high_semantics:
        units_dict["18"] = [2, 2, 2, 1, 1]
    units = units_dict[str(depth)]
    if depth >= 50:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    filter_list = [x*channel_factor for x in filter_list]
    num_stage = len(units)
    # append if filter_list is not enough
    if num_stage >= len(filter_list):
        append_len = len(filter_list) - num_stage + 1
        for i in range(append_len):
            filter_list.append(filter_list[-1]*2)
        print("Appended filter list: {}".format(str(filter_list)))
