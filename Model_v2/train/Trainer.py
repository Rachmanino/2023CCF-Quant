import time

import numpy as np
import torch.optim
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import MultiStepLR as Scheduler



from config import Config
from nets.tcn import Tcn_Module
from train.batch_loss import Batch_Loss
from utils import get_loaders, get_logger


class  Trainer():
    def __init__(self, config:Config):
        self.learning_rate =config.learning_rat
        self.max_epochs = config.epochs
        self.device = config.device
        self.model = Tcn_Module(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate)
        self.scheduler = Scheduler(self.optimizer, milestones=[20,50,100], gamma=0.1)

        self.clip_grad_norm = config.clip_grad_norm
        self.batch_loss = Batch_Loss(config)
        self.tensorboard = SummaryWriter(f"{config.cwd}/tensorboard")

        self.train_dataloader_list,  self.valid_dataloader_list= get_loaders(config=config)

        self.logger = get_logger(config.log_file)

        self.epochs= 1
        self.best_val = 0

        '''model save parameter'''
        self.if_over_write = config.if_over_write
        self.save_gap = config.save_gap
        self.save_counter = 0
        self.cwd = config.cwd
        self._logging_parameter(config)







    def run(self):

        epoch_train = []
        epoch_valid = []
        epoch_index = []
        for epoch in range(self.epochs, self.max_epochs+1):
            self.logger.info('=================================================================')
            start_time = time.time()

            # LR Decay

            ###Training
            train_loss, valid_loss = self._train_one_epoch()
            self.scheduler.step()
            epoch_train.append(train_loss)
            epoch_valid.append(valid_loss)
            epoch_index.append(epoch)
            end_time = time.time()
            elapsed_time = start_time - end_time
            self.logger.info("Epoch {:3d}/{:3d}: Time: {:.2f}]".format(
                epoch, self.max_epochs, elapsed_time / 60))

            ####save model
            prev_max_val = self.best_val
            self.best_val = max(self.best_val, -valid_loss)

            if_save = -valid_loss > prev_max_val
            self.save_counter += 1
            model_path = None
            if if_save:
                if self.if_over_write:
                    model_path = f"{self.cwd}/model.pt"
                else:
                    model_path = f"{self.cwd}/model__{self.epochs:04}_{self.best_val:.4f}.pt"
            elif self.save_counter >= self.save_gap:
                self.save_counter =0
                # if self.if_over_write:
                #     model_path = f"{self.cwd}/model.pt"
                # else:
                model_path = f"{self.cwd}/model__{self.epochs:04}_{self.best_val:.4f}.pt"

            if model_path:
                torch.save(self.model, model_path)


        self.logger.info("PLOT THE RESULT")
        fig_path =f"{self.cwd}/result.png"
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epoch_index, epoch_train, label="train")
        plt.plot(epoch_index, epoch_valid, label="valid")
        plt.legend()
        plt.savefig(fig_path)
        self.logger.info("Train End!!!!!!")
        torch.save(self.model,f"{self.cwd}/model_end.pt")




    def _logging_parameter(self, config:Config):
        self.logger.info('Start Training, the hyperparameters are shown below:')
        self.logger.info("*************************************************")
        self.logger.info('--save_dir:   {}'.format(config.cwd))
        self.logger.info('--batch_size:   {}'.format(config.batch_size))
        self.logger.info('--day_windows:   {}'.format(config.day_windows))
        self.logger.info('--epochs:   {}'.format(config.epochs))
        self.logger.info('--learning_rate:   {}'.format(config.learning_rate))
        self.logger.info('--gpu_id:   {}'.format(config.device))
        self.logger.info('--drop:   {}'.format( config.drop))






    def _train_one_epoch(self):
        self.model.train()
        train_epoch_loss = []
        for train_dataloader in self.train_dataloader_list:
            self.logger.info("*************************************************"+train_dataloader.dataset.name)
            for idx, (data, embedding_intput , targets) in enumerate(tqdm(train_dataloader)):
                # data = data.to(self.device)
                # targets = targets.to(self.device)
                preds = self.model(data, embedding_intput)


                ####loss.shape == [batch, num_indicator]
                loss = self.batch_loss(preds,targets)
                # loss = torch.abs(loss).mean()
                loss = loss.mean()
                ###minize
                self.optimizer_update(loss)
                # loss = -loss
                train_epoch_loss.append(loss.cpu().detach().numpy())
            torch.cuda.empty_cache()
        mean_train_loss = np.stack(train_epoch_loss).mean()

        with torch.no_grad():
            self.model.eval()
            val_epoch_loss = []
            for valid_dataloader in  self.valid_dataloader_list:
                for idx, (data, embedding_intput, targets) in enumerate(tqdm(valid_dataloader)):
                    # data = data.to(self.device)
                    # targets = targets.to(self.device)
                    preds = self.model(data, embedding_intput)

                    ####loss.shape == [batch, num_indicator]
                    loss = self.batch_loss(preds, targets)
                    loss = loss.mean()
                    ###minize
                    # loss = -loss
                    val_epoch_loss.append(loss.cpu().detach().numpy())
            mean_val_loss = np.stack(val_epoch_loss).mean()
        self.logger.info('Epoch {:3d}: Train Loss: ({:.4f}) ,  Val Loss: {:.4f}'
                         .format(self.epochs, mean_train_loss, mean_val_loss))

        self.tensorboard.add_scalar("train", -mean_train_loss)
        self.tensorboard.add_scalar("valid", -mean_val_loss)
        self.epochs += 1

        return mean_train_loss, mean_val_loss

    def optimizer_update(self,  objective: Tensor):
        self.optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=self.optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        self.optimizer.step()






