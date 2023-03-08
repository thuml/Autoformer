import os
import torch
from exp.exp_main import Exp_Main
import random
import pandas as pd
import numpy as np
from utils.tools import dotdict
from datetime import timedelta, datetime
from pathlib import Path


#%%
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args = dotdict()
args.target = 'OT'  #
args.des = 'Exp'  #
args.dropout = 0.0  #
args.num_workers = 1  #
args.gpu = 0  #
args.lradj = 'type1'  #
args.devices = '0'  #
args.use_gpu = False  #
args.use_multi_gpu = False  #
# if args.use_gpu and args.use_multi_gpu: #是否使用多卡的判断
#     args.dvices = args.devices.replace(' ', '')
#     device_ids = args.devices.split(',')
#     args.device_ids = [int(id_) for id_ in device_ids]
#     args.gpu = args.device_ids[0]
args.freq = 't'  #
args.checkpoints = './checkpoints/'  #
args.bucket_size = 4
args.n_hashes = 4
args.is_training = True  #
args.root_path = './dataset/'  # TODO
args.data_path = 'data.csv'  # TODO
args.model_id = 'Traffic_Mini_Test'  #
args.model = 'Transformer'  #
args.data = 'custom'  #
args.features = 'M'  #
args.seq_len = 1  #
args.label_len = 0  #
args.pred_len = 1  #
args.e_layers = 2  #
args.d_layers = 1  #
args.n_heads = 8  #
args.factor = 1  #
args.enc_in = 3  #
args.dec_in = 1  #
args.c_out = 1  #
args.d_model = 512  #
args.des = 'Exp'  #
args.itr = 1  #
args.d_ff = 2048  #
args.moving_avg = 1  #
args.distil = True  #
args.output_attention = True  #
args.patience = 10  #
args.learning_rate = 0.0001  #
args.batch_size = 1  #
args.embed = 'timeF'  #
args.activation = 'gelu'  #
args.use_amp = False  #
args.loss = 'mse'  #
args.train_epochs = 1  #

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
device_ids = args.devices.split(',')
args.device_ids = [int(id_) for id_ in device_ids]
args.gpu = args.device_ids[0]

# print('Args in experiment:')
# print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()