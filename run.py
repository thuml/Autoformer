import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

import wandb
from datetime import datetime


def main():

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--seed', type=int, required=True, default=0, help='randomization seed')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')
    
    # logging
    # eval steps
    parser.add_argument('--eval_steps', type=int, default=440, help='Test every eval_steps')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, help='root path of the data file')
    parser.add_argument('--data_path', type=str, help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # wandb
    parser.add_argument('--wandb_run', type=str, default='missing name', help='wandb run')
    parser.add_argument('--wandb_project', type=str, default='Autoformer', help='wandb project')
    parser.add_argument('--experiment_tag', type=str, default='e0_untagged_experiment', help='wandb project')

    # Constrained
    parser.add_argument('--constraint_type', type=str, help='Constraint type (erm,constant,static_linear,dynamic_linear,resilience)')
    parser.add_argument('--constraint_level', type=float, help='Constraint level (epsilon) if using constant constraint_type')    
    parser.add_argument('--constraint_slope', type=float, help='Constraint slope if using static_linear or dynamic_linear')
    parser.add_argument('--constraint_offset', type=float, help='Constraint offset if using static_linear or dynamic_linear')
    parser.add_argument('--dual_lr',  type=float, help='dual learning rate')
    parser.add_argument('--dual_init',  type=float, help='dual var initialization')
    parser.add_argument('--dual_clip',  type=float, default=10.0, help='clip dual variables')
    parser.add_argument('--sampling', action='store_true', default=False, help='Wether sample time steps in Lagrangian')

    # Resilient
    parser.add_argument('--resilient_lr', type=float, default=0.0, help='Resilient learning rate')
    parser.add_argument('--resilient_cost_alpha', type=float, default=2.0, help='resilient quadratic cost penalty')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    args = parser.parse_args()

    # Argument validation
    # if Constant, then constraint_level must be provided
    if args.constraint_type == 'constant' and args.constraint_level is None:
        raise ValueError("Constraint type is constant, but constraint_level is None")
    # if StaticLinear or DynamicLinear, then constraint_slope and constraint_offset must be provided
    if args.constraint_type in ['static_linear','dynamic_linear'] and (args.constraint_slope is None or args.constraint_offset is None):
        raise ValueError("Constraint type is static_linear or dynamic_linear, but constraint_slope or constraint_offset is None")
    # if not ERM, then dual_lr and dual_init must be provided
    if args.constraint_type != 'erm' and (args.dual_lr is None or args.dual_init is None):
        raise ValueError("Constraint type is not erm, but dual_lr or dual_init is None")

    if args.seed==0:
       print("No seed provided (--seed 0), using current time as seed")
       print("Randomly generating seed from time: ")
       args.seed = int(datetime.now().timestamp())
    else: 
        print(f"Using user provided seed")
    print(f"Seed is {args.seed}, this will be reflected in wandb config.")
    
    #cast seed as int 
    # seed = int(args.seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    

    print("Starting run with args")
    print(args)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    run_name = f"{args.wandb_run}/{args.data_path}_{args.model}_len{args.pred_len}"
    wandb.init(name=run_name, project=args.wandb_project, config=args,tags=[args.experiment_tag])
    # Real seed is the actual seed passed to the RNGs (The user might have passed seed=0 to autogenerate seeds)
    wandb.log({"real_seed":args.seed})

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_constr_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_wb_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.constraint_type,
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
                args.des, 
                ii,
                # uuid fron wandb run
                wandb.run.id,
                )
            # log the file ID used for writing the test preds.
            wandb.log({"file_id":setting})

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


if __name__ == "__main__":
    main()