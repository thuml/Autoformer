
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer, Linear,DLinear,PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import wandb

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        if self.args.use_amp:
            raise Exception("This repo does not support AMP. ")

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Linear': Linear,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        #criterion = nn.MSELoss()
        return lambda x, y: ((x-y)**2).mean(dim=(0, 2))#criterion

    def vali(self, vali_data, vali_loader, criterion):
        "Javier: this returns overall mean loss, average losses per step, and overall average metrics."
        total_loss = []
        total_losses = []
        total_metrics=[]
        total_infeasibilities = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                # Compute metrics (detach and convert to numpy)
                mae, mse, rmse, mape, mspe = metric(pred=outputs.detach().cpu().numpy(), true=batch_y.detach().cpu().numpy())

                loss = criterion(pred, true)
                
                #TODO verify this logic and reenable. 
                vali_num_infeasibles = (loss > self.args.constraint_level).sum()
                #vali_infeasible_rate = vali_num_infeasibles / self.args.pred_len

                #print(f"Number of infeasibilities: {vali_num_infeasibles}/{self.args.pred_len} rate {vali_infeasible_rate}")

                total_loss.append(loss.mean().item())
                total_losses.append(loss.cpu().numpy())
                total_metrics.append([mae, mse, rmse, mape, mspe])
                total_infeasibilities.append(vali_num_infeasibles)

        total_loss = np.average(total_loss)
        total_losses = np.stack(total_losses)
        total_metrics = np.stack(total_metrics)
        total_infeasibilities = np.average(total_infeasibilities)
        average_infeasiblity_rate = total_infeasibilities / self.args.pred_len

        # nasty way to pass all metrics together.
        metrics = tuple(total_metrics.mean(axis=0))
        self.model.train()
        return total_loss, total_losses.mean(axis=0), metrics, total_infeasibilities, average_infeasiblity_rate

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        multipliers = torch.ones(self.args.pred_len, device=self.device)*self.args.dual_init
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # TODO: PatchTST uses this. Do we want it? 
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)


        early_stopped_before=False # we want to log when is early stopping triggered.
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_losses = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input: pred_len labeled examples, then a bunch of zeros the size of the pred len.
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder 
            
                if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_all = criterion(outputs, batch_y)
                    

                    # Calculate other metrics for logging (detach and convert to numpy)
                    train_mae, train_mse, train_mse, train_mape, train_mspe = metric(
                        pred=outputs.detach().cpu().numpy(), 
                        true=batch_y.detach().cpu().numpy()
                    )

                    train_losses.append(loss_all.cpu().detach())
                    train_loss.append(loss_all.mean().item())

                    #loss = (loss_all*multipliers).sum() #old loss
                    if self.args.constrained_learning: 
                        constrained_loss, multipliers = self._compute_constrained_loss(loss_all, multipliers, self.args)
                    if self.args.dual_lr>0:
                        constrained_loss = ((multipliers + 1/self.args.pred_len) * loss_all).sum()
                        multipliers = (multipliers+self.args.dual_lr*(loss_all.detach()-self.args.constraint_level)).clamp(min=0.)
                    else:
                        constrained_loss = loss_all.mean()
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, constrained_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                constrained_loss.backward()
                model_optim.step()
            
                # New (PatchTST)
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                if (i + 1) % self.args.eval_steps == 0:
                    #TODO address duplicates between this and end of epoch.
                    # Calculating how many violate feasibility
                    # TODO WARNING this will only work in the constant case.
                    train_num_infeasibles = (loss_all > self.args.constraint_level).sum()
                    train_infeasible_rate = train_num_infeasibles / self.args.pred_len

                    print(f"Number of infeasibilities: {train_num_infeasibles}/{self.args.pred_len} rate {train_infeasible_rate}")

                    #Run validation set
                    vali_loss, vali_losses, val_metrics, val_infeasibilities, val_avg_infeasiblity_rate = self.vali(vali_data, vali_loader, criterion)
                    test_loss, test_losses, test_metrics, test_infeasibilities, test_avg_infeasiblity_rate= self.vali(test_data, test_loader, criterion)

                    val_mae, _, val_rmse, val_mape, val_mspe = val_metrics
                    test_mae, _, test_rmse, test_mape, test_mspe = test_metrics

                    avg_train_loss = np.average(train_loss)
                    #TODO ADD WANDB LOG !!!!
                    wandb.log(
                        {   
                            # train
                            "mse/train": avg_train_loss,
                            "mse/val": vali_loss,
                            "mse/test": test_loss,
                            
                            # val metrics
                            "mae/val": val_mae,
                            "rmse/val": val_rmse,
                            "mape/val": val_mape,
                            "mspe/val": val_mspe,

                            # test metrics
                            "mae/test": test_mae,
                            "rmse/test": test_rmse,
                            "mape/test": test_mape,
                            "mspe/test": test_mspe,

                            # infeasibles
                            "infeasibles/train": train_num_infeasibles, 
                            "infeasible_rate/train": train_infeasible_rate, 
                            "infeasibles/val": val_infeasibilities,
                            "infeasible_rate/val": val_avg_infeasiblity_rate,
                            "infeasibles/test": test_infeasibilities,
                            "infeasible_rate/test": test_avg_infeasiblity_rate,

                            "epoch":epoch+1,
                        },
                        commit=False
                    )

                wandb.log(
                        {   
                            "loss": constrained_loss.detach().cpu().item(),
                            "mae/train": train_mae,
                            "mse/train": train_mse,
                            "rmse/train": train_mse,
                            "mape/train": train_mape,
                            "mspe/train": train_mspe,
                        }
                )   
            ## =======END EPOCH LOOP

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            avg_epoch_train_loss = np.average(train_loss)
            train_losses = np.stack(train_losses)
            train_losses = np.average(train_losses, axis=0) #This is losses per step

            train_num_infeasibles = (loss_all > self.args.constraint_level).sum()
            train_infeasible_rate = train_num_infeasibles / self.args.pred_len

            print(f"Number of infeasibilities: {train_num_infeasibles}/{self.args.pred_len} rate {train_infeasible_rate}")

            # Run validation again for end of epoch logging
            vali_loss, vali_losses, val_metrics, val_infeasibilities, val_avg_infeasiblity_rate = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_losses, test_metrics, test_infeasibilities, test_avg_infeasiblity_rate= self.vali(test_data, test_loader, criterion)

            val_mae, _, val_rmse, val_mape, val_mspe = val_metrics
            test_mae, _, test_rmse, test_mape, test_mspe = test_metrics

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, avg_epoch_train_loss, vali_loss, test_loss))
            print(f"Train Loss[:10]: {train_losses[:10]}")
            print(f"Val Loss[:10]: {vali_losses[:10]}")
            print(f"Test Loss[:10]: {test_losses[:10]}")
            print(f"Train Loss[-10:]: {train_losses[-10:]}")
            print(f"Val Loss[-10:]: {vali_losses[-10:]}")
            print(f"Test Loss[-10:]: {test_losses[-10:]}")
    
            
            for split, losses in zip(["train", "val", "test"],[train_losses, vali_losses, test_losses]):
                for i, constrained_loss in enumerate(losses):
                    wandb.log({f"mse/{split}/{i}": constrained_loss, "epoch":epoch+1},commit=False)

            for i, multiplier in enumerate(multipliers):
                wandb.log({f"multiplier/{i}": multiplier, "epoch":epoch+1},commit=False)

            early_stopping(vali_loss, self.model, path) #must keep this even if we don't early stop, to save best model.
            if early_stopping.early_stop and not early_stopped_before:
                print(f"Early stopping triggered at epoch {epoch+1}. Will continue training.")
                wandb.log({"early_stopped_epoch":epoch+1, "epoch":epoch+1},commit=False)
                early_stopped_before=True
            
            if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            
            # Log all metrics together
            wandb.log(
                {   
                    # train
                    "mse/train": avg_epoch_train_loss,
                    "mse/val": vali_loss,
                    "mse/test": test_loss,
                    
                    # val metrics
                    "mae/val": val_mae,
                    "rmse/val": val_rmse,
                    "mape/val": val_mape,
                    "mspe/val": val_mspe,

                    # test metrics
                    "mae/test": test_mae,
                    "rmse/test": test_rmse,
                    "mape/test": test_mape,
                    "mspe/test": test_mspe,

                    # infeasibles
                    "infeasibles/train": train_num_infeasibles,
                    "infeasible_rate/train": train_infeasible_rate,
                    "infeasibles/val": val_infeasibilities,
                    "infeasible_rate/val": val_avg_infeasiblity_rate,
                    "infeasibles/test": test_infeasibilities,
                    "infeasible_rate/test": test_avg_infeasiblity_rate,
                    "epoch":epoch+1,
                },
                commit=True
            )

            # updated from patchTST
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        ##===END TRAINING LOOP

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
