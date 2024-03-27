
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer, Linear,DLinear,PatchTST,Koopa,Pyraformer,FEDformer,Nonstationary_Transformer,TimesNet, MICN, FiLM
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import scipy
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas

import wandb

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        if self.args.use_amp:
            raise Exception("This repo does not support AMP. ")

    #to be able to load back the modelb
    MODEL_DICT={
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Linear': Linear,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'Koopa': Koopa,
            'Pyraformer': Pyraformer,
            'FEDformer': FEDformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'TimesNet': TimesNet,
            'MICN': MICN,
            'FiLM': FiLM,
    }
    
    def _get_mask_spectrum(self):
        """
        get shared frequency spectrums
        """
        train_data, train_loader = self._get_data(flag='train')
        amps = 0.0
        for data in train_loader:
            lookback_window = data[0]
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)

        mask_spectrum = amps.topk(int(amps.shape[0]*self.args.alpha)).indices
        return mask_spectrum # as the spectrums of time-invariant component

    def _build_model(self):
        model_dict = self.MODEL_DICT
        if self.args.model == 'Koopa':
            self.args.mask_spectrum = self._get_mask_spectrum()
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
    
    def _create_constraint_levels_tensor(self):
        """Creates a constraint_levels tensor of shape (pred_len,) for use in constraint optimization."""
        # Setting vector of constraints if running linear or constant constrained.
        device = self.device
        if self.args.constraint_type == "static_linear" or self.args.constraint_type == "dynamic_linear":
            constraint_levels = (torch.arange(self.args.pred_len)*self.args.constraint_slope+ self.args.constraint_offset).to(device)
        elif self.args.constraint_type == "constant" or self.args.constraint_type == "resilience":
            #TODO run and test that dimensions match the above.
            constraint_levels = (torch.ones(self.args.pred_len, device=device)*self.args.constraint_level).to(device)
        elif self.args.constraint_type == "erm" or self.args.constraint_type == "monotonic":
            constraint_levels = torch.zeros(self.args.pred_len, device=device).to(device)
        else: 
            #raise ValueError(f"{self.args.constraint_type} Constraint type not implemented yet.")
            print(f"WARNING RUNNING WITH UNSUPPORTED CONSTRAINT TYPE {self.args.constraint_type}!!!!")
        #TODO add monotonic constraint levels. 
        #return constraint_levels
        return constraint_levels.detach() #Don't really need gradients for it

    def _select_criterion(self):
        return lambda x, y: ((x-y)**2).mean(dim=(0, 2))#criterion
    
    def _rename_dict(self,d, suffix):
                        return {f"{key}/{suffix}": val for key, val in d.items()}

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Initializing multipliers for constraint optimization]
        multipliers = torch.ones(self.args.pred_len-(self.args. constraint_type == "monotonic"), device=self.device)*self.args.dual_init
        slacks = torch.zeros(self.args.pred_len-(self.args.constraint_type == "monotonic"), device=self.device)
        
        # Setting vector of constraints if running linear or constant constrained.
        constraint_levels = self._create_constraint_levels_tensor()
        
        for i, eps in enumerate(constraint_levels):
                wandb.log({f"constraint/{i}": eps},commit=False)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            if len(os.path.basename(path))>50:
                path = os.path.join(os.path.dirname(path), os.path.basename(path)[-50:])
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = None 
        if self.args.lradj =='TST':
            print("Creating PatchTST scheduler object")
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        else:
            scheduler = None


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
                raw_loss = criterion(outputs, batch_y)
                detached_raw_loss = raw_loss.detach()

                # Calculate other metrics for logging (detach and convert to numpy)
                train_mae, train_mse, train_rmse, train_mape, train_mspe = metric(
                    pred=outputs.detach().cpu().numpy(), 
                    true=batch_y.detach().cpu().numpy()
                )

                train_losses.append(raw_loss.cpu().detach())
                train_loss.append(raw_loss.mean().item())

                # Constraint optimization
                if self.args.constraint_type == "erm":
                    constrained_loss = raw_loss.mean()
                elif self.args.constraint_type == "constant" or self.args.constraint_type == "static_linear" or self.args.constraint_type =="resilience":
                    if not self.args.sampling:
                        constrained_loss = ((multipliers + 1/self.args.pred_len) * raw_loss).sum()
                    else:
                        probabilities = (multipliers + 1/self.args.pred_len).unsqueeze(0).repeat(batch_x.shape[0],1)
                        sampled_indexes = torch.multinomial(probabilities, self.args.pred_len, replacement=True)
                        constrained_loss = raw_loss[sampled_indexes].mean()
                    #TODO uncomment for dual restarts
                    #multipliers = multipliers * (raw_loss > constraint_levels).float()
                    multipliers += self.args.dual_lr * (detached_raw_loss - (constraint_levels+slacks))
                    multipliers = torch.clip(multipliers, 0.0, self.args.dual_clip)
                elif self.args.constraint_type == "monotonic":
                    constrained_loss = ((multipliers[:-1] + multipliers[1:] - 1/self.args.pred_len) * raw_loss[1:-1]).sum()
                    constrained_loss += (1/self.args.pred_len+multipliers[0]) * raw_loss[0] 
                    constrained_loss += (1/self.args.pred_len-multipliers[-1]) * raw_loss[-1] 
                    
                    multipliers += self.args.dual_lr * (detached_raw_loss[:-1]-detached_raw_loss[1:]-(constraint_levels[1:]+slacks))
                    multipliers = torch.clip(multipliers, 0.0, self.args.dual_clip)
                elif self.args.constraint_type == "dynamic_linear":
                    raise NotImplementedError("dynamic_linear constraint not implemented yet.")
                else:
                    raise ValueError(f"{self.args.constraint_type} Constraint type not implemented yet.")
                
                #If resilience activated
                if self.args.resilient_lr > 0:
                    slacks += self.args.resilient_lr * (-self.args.resilient_cost_alpha * slacks + multipliers)
                    slacks = torch.clip(slacks, min=0.0)
                constrained_loss.backward()
                model_optim.step()
                
                #Logging: print every 100 iterations
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, constrained_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                    # New (PatchTST)
                    if self.args.lradj == 'TST':
                        adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                        scheduler.step()

                    if (i + 1) % self.args.eval_steps == 0:
                        #TODO address duplicates between this and end of epoch.
                        # Calculating how many violate feasibility
                        # TODO WARNING this will only work in the constant case.
                        if self.args.constraint_type == "monotonic":
                            train_num_infeasibles = (detached_raw_loss[1:] > detached_raw_loss[:-1]).sum()
                        else:
                            train_num_infeasibles = (detached_raw_loss > (constraint_levels+slacks)).sum()
                        train_infeasible_rate = train_num_infeasibles / self.args.pred_len

                        print(f"Number of infeasibilities: {train_num_infeasibles}/{self.args.pred_len} rate {train_infeasible_rate}")

                        #Run validation set
                        vali_loss, vali_losses, val_metrics, val_infeasibilities, val_avg_infeasiblity_rate, val_loss_distr_metrics, val_loss_distr_metrics_per_timestep = self.vali(vali_data, vali_loader, criterion)
                        
                        # Append /val to the val_loss_distr_metrics
                        val_loss_distr_metrics = self._rename_dict(val_loss_distr_metrics, "val")
                        val_loss_distr_metrics_per_timestep = self._rename_dict(val_loss_distr_metrics_per_timestep, "val")

                        test_loss, test_losses, test_metrics, test_infeasibilities, test_avg_infeasiblity_rate, test_loss_distr_metrics, test_loss_distr_metrics_per_timestep = self.vali(test_data, test_loader, criterion)
                        # Append /test to the test_loss_distr_metrics
                        test_loss_distr_metrics = self._rename_dict(test_loss_distr_metrics, "test")
                        test_loss_distr_metrics_per_timestep = self._rename_dict(test_loss_distr_metrics_per_timestep, "test")

                        val_mae, _, val_rmse, val_mape, val_mspe = val_metrics
                        test_mae, _, test_rmse, test_mape, test_mspe = test_metrics
                        
                        steps = torch.arange(self.args.pred_len).detach().cpu().numpy()

                        train_linearity = pearsonr(steps,detached_raw_loss.cpu().numpy())[0]
                        vali_linearity = pearsonr(steps,vali_losses)[0]
                        test_linearity = pearsonr(steps,test_losses)[0]

                        avg_train_loss = np.average(train_loss)
                        
                        # Intra epoch logging
                        wandb.log(
                            {   
                                # train
                                "mse/train": avg_train_loss,
                                "linearity/train": train_linearity,
                                "mse/val": vali_loss,
                                "mse/test": test_loss,
                                
                                # val metrics
                                "mae/val": val_mae,
                                "rmse/val": val_rmse,
                                "mape/val": val_mape,
                                "mspe/val": val_mspe,
                                "linearity/val": vali_linearity,

                                # test metrics
                                "mae/test": test_mae,
                                "rmse/test": test_rmse,
                                "mape/test": test_mape,
                                "mspe/test": test_mspe,
                                "linearity/test": test_linearity,

                                # infeasibles
                                "infeasibles/train": train_num_infeasibles, 
                                "infeasible_rate/train": train_infeasible_rate, 
                                "infeasibles/val": val_infeasibilities,
                                "infeasible_rate/val": val_avg_infeasiblity_rate,
                                "infeasibles/test": test_infeasibilities,
                                "infeasible_rate/test": test_avg_infeasiblity_rate,
                                "epoch":epoch+1,

                                # loss distribution metrics
                                **val_loss_distr_metrics,
                                **test_loss_distr_metrics,
                            },
                            commit=False
                        )
                    # End of batch logging
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

            if self.args.constraint_type == "monotonic":
                train_num_infeasibles = (detached_raw_loss[1:] > detached_raw_loss[:-1]).sum()
            else:
                train_num_infeasibles = (detached_raw_loss > (constraint_levels+slacks)).sum()
            train_infeasible_rate = train_num_infeasibles / self.args.pred_len

            print(f"Number of infeasibilities: {train_num_infeasibles}/{self.args.pred_len} rate {train_infeasible_rate}")

            #Run validation set again for epoch logging. TODO remove code duplication.
            vali_loss, vali_losses, val_metrics, val_infeasibilities, val_avg_infeasiblity_rate, val_loss_distr_metrics, val_loss_distr_metrics_per_timestep = self.vali(vali_data, vali_loader, criterion)
            
            # Append /val to the val_loss_distr_metrics
            val_loss_distr_metrics = self._rename_dict(val_loss_distr_metrics, "val")
            val_loss_distr_metrics_per_timestep = self._rename_dict(val_loss_distr_metrics_per_timestep, "val")

            test_loss, test_losses, test_metrics, test_infeasibilities, test_avg_infeasiblity_rate, test_loss_distr_metrics, test_loss_distr_metrics_per_timestep = self.vali(test_data, test_loader, criterion)
            # Append /test to the test_loss_distr_metrics
            test_loss_distr_metrics = self._rename_dict(test_loss_distr_metrics, "test")
            test_loss_distr_metrics_per_timestep = self._rename_dict(test_loss_distr_metrics_per_timestep, "test")

            val_mae, _, val_rmse, val_mape, val_mspe = val_metrics
            test_mae, _, test_rmse, test_mape, test_mspe = test_metrics

            steps = torch.arange(self.args.pred_len).detach().cpu().numpy()
            train_linearity = pearsonr(steps,raw_loss.detach().cpu().numpy())[0]
            vali_linearity = pearsonr(steps,vali_losses)[0]
            test_linearity = pearsonr(steps,test_losses)[0]

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, avg_epoch_train_loss, vali_loss, test_loss))
            print(f"Train Loss[:10]: {train_losses[:10]}")
            print(f"Val Loss[:10]: {vali_losses[:10]}")
            print(f"Test Loss[:10]: {test_losses[:10]}")
            print(f"Train Loss[-10:]: {train_losses[-10:]}")
            print(f"Val Loss[-10:]: {vali_losses[-10:]}")
            print(f"Test Loss[-10:]: {test_losses[-10:]}")
    
            # Logging train, val, test losses
            for split, losses in zip(["train", "val", "test"],[train_losses, vali_losses, test_losses]):
                for i, loss in enumerate(losses):
                    wandb.log({f"mse/{split}/{i}": loss, "epoch":epoch+1},commit=False)
            
            # Logging constraint optimization info info
            if multipliers is not None:
                for i, multiplier in enumerate(multipliers):
                    wandb.log({f"multiplier/{i}": multiplier, "epoch":epoch+1},commit=False)
            if self.args.resilient_lr>0:
                for i, slack in enumerate(slacks):
                    wandb.log({f"slack/{i}": slack, "epoch":epoch+1},commit=False)
            
            # Logging loss distribution metrics per timestep. No need to zip val test
            for loss_distr_metrics in [val_loss_distr_metrics_per_timestep, test_loss_distr_metrics_per_timestep]:
                for metric_name, metric_value in loss_distr_metrics.items():
                    for i, value in enumerate(metric_value):
                        wandb.log({f"{metric_name}/{i}": value, "epoch":epoch+1},commit=False)


            # Early stopping, checkpointing, LR adj
            early_stopping(vali_loss, self.model, path) #must keep this even if we don't early stop, to save best model.
            # if early_stopping.early_stop:
            #     print(f"Early stopping at epoch {epoch}")
            #     break
            if early_stopping.early_stop and not early_stopped_before:
                print(f"Early stopping triggered at epoch {epoch+1}. Will continue training.")
                wandb.log({"early_stopped_epoch":epoch+1, "epoch":epoch+1},commit=False)
                early_stopped_before=True
            
            if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            
            #End of epoch logging
            # Log all metrics together
            wandb.log(
                {   
                    # train
                    "mse/train": avg_epoch_train_loss,
                    "mse/val": vali_loss,
                    "mse/test": test_loss,
                    "linearity/train": train_linearity,
                    
                    # val metrics
                    "mae/val": val_mae,
                    "rmse/val": val_rmse,
                    "mape/val": val_mape,
                    "mspe/val": val_mspe,
                    "linearity/val": vali_linearity,

                    # test metrics
                    "mae/test": test_mae,
                    "rmse/test": test_rmse,
                    "mape/test": test_mape,
                    "mspe/test": test_mspe,
                    "linearity/test": test_linearity,

                    # infeasibles
                    "infeasibles/train": train_num_infeasibles,
                    "infeasible_rate/train": train_infeasible_rate,
                    "infeasibles/val": val_infeasibilities,
                    "infeasible_rate/val": val_avg_infeasiblity_rate,
                    "infeasibles/test": test_infeasibilities,
                    "infeasible_rate/test": test_avg_infeasiblity_rate,
                    "epoch":epoch+1,

                    # loss distribution metrics
                    **val_loss_distr_metrics,
                    **test_loss_distr_metrics,
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

        # Save to "./checkpoints/best_model.pth"
        best_model_path_final=os.path.join(wandb.run.dir, 'best_model.pth')
        print("Logging the best model to ", best_model_path_final)
        torch.save(self.model.state_dict(),best_model_path_final)
        wandb.save(best_model_path_final,policy='now')

        return

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_losses = []
        total_metrics=[]
        total_infeasibilities = []
        unaggregated_losses = []

        constraint_levels = self._create_constraint_levels_tensor()

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

                pred = outputs.detach()
                true = batch_y.detach()

                # Compute metrics (detach and convert to numpy)
                mae, mse, rmse, mape, mspe = metric(pred=outputs.detach().cpu().numpy(), true=batch_y.detach().cpu().numpy())
                
                loss = criterion(pred, true)

                # Shape (batch_size, pred_len, out_dim). Average losses over out_dim.
                all_losses_per_window_example=((pred-true)**2).mean(dim=(2))
                
                vali_num_infeasibles = (loss > (constraint_levels)).sum()
                if self.args.constraint_type == "monotonic":
                    vali_num_infeasibles = (loss[1:] > loss[:-1]).sum()
                else:
                    vali_num_infeasibles = (loss > (constraint_levels)).sum()
                #vali_infeasible_rate = vali_num_infeasibles / self.args.pred_len

                    #print(f"Number of infeasibilities: {vali_num_infeasibles}/{self.args.pred_len} rate {vali_infeasible_rate}")
                unaggregated_losses.append(all_losses_per_window_example.detach().cpu().numpy())
                total_loss.append(loss.mean().item())
                total_losses.append(loss.cpu().numpy())
                total_metrics.append([mae, mse, rmse, mape, mspe])
                total_infeasibilities.append(vali_num_infeasibles.cpu().numpy())

        total_loss = np.average(total_loss)
        total_losses = np.stack(total_losses)
        total_metrics = np.stack(total_metrics)
        total_infeasibilities = np.average(total_infeasibilities)
        average_infeasiblity_rate = total_infeasibilities / self.args.pred_len

        unaggregated_losses = np.concatenate(unaggregated_losses, axis=0)

        # Compute descriptive statistics (IQR, median,max,min) over all examples and timesteps
        pct_25 = np.percentile(unaggregated_losses, 25)
        pct_50 = np.percentile(unaggregated_losses, 50)
        pct_75 = np.percentile(unaggregated_losses, 75)
        pct_95 = np.percentile(unaggregated_losses, 95) #var_risk
        pct_99 = np.percentile(unaggregated_losses, 99) #var_risk
        std = np.std(unaggregated_losses)
        max = np.max(unaggregated_losses)

        # Per timestep statistics
        pct_25_per_timestep = np.percentile(unaggregated_losses, 25, axis=0)
        pct_50_per_timestep = np.percentile(unaggregated_losses, 50, axis=0)
        pct_75_per_timestep = np.percentile(unaggregated_losses, 75, axis=0)
        pct_95_per_timestep = np.percentile(unaggregated_losses, 95, axis=0) #var_risk
        pct_99_per_timestep = np.percentile(unaggregated_losses, 99, axis=0) #var_risk
        std_per_timestep = np.std(unaggregated_losses, axis=0)
        max_per_timestep = np.max(unaggregated_losses, axis=0)

        # Loss distribution metrics
        loss_distribution_metrics = {
            "pct_25":pct_25,
            "pct_50":pct_50,
            "pct_75":pct_75,
            "pct_95":pct_95,
            "pct_99":pct_99,
            "std":std,
            "max":max,
        }
        loss_distribution_metrics_per_timestep = {
            "pct_25_per_timestep":pct_25_per_timestep,
            "pct_50_per_timestep":pct_50_per_timestep,
            "pct_75_per_timestep":pct_75_per_timestep,
            "pct_95_per_timestep":pct_95_per_timestep,
            "pct_99_per_timestep":pct_99_per_timestep,
            "std_per_timestep":std_per_timestep,
            "max_per_timestep":max_per_timestep,
        }

        # nasty way to pass all metrics together.
        metrics = tuple(total_metrics.mean(axis=0))
        self.model.train()
        return total_loss, total_losses.mean(axis=0), metrics, total_infeasibilities, average_infeasiblity_rate,loss_distribution_metrics,loss_distribution_metrics_per_timestep

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        if len(setting)>30:
            folder_path = './test_results/' + setting[-30:] + '/'
        else:
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

        # In case the artifact doesn't work
        # wandb.log({"test_predictions": preds})
        # wandb.log({"test_true": trues})

        # Log a table with preds and trues
        # test_df = pandas.DataFrame(data=np.concatenate([preds, trues], axis=1), columns=[f"pred_{i}" for i in range(preds.shape[1])] + [f"true_{i}" for i in range(trues.shape[1])])
        # test_df.to_csv("deleteme.csv")
        # wandb.save("pred.npy")
        # wandb.log({"test_table": wandb.Table(dataframe=test_df)})

        #in case the log doesn't work
        # artifact = wandb.Artifact("predictions", type="dataset")
        # artifact.add_dir(f"{folder_path}/")
        # wandb.log_artifact(artifact)
        
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
