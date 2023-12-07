import os

from torch.utils.data import DataLoader


import logging

from config import Config
from dataset.stock_dataset import Stock_Dataset


def get_loaders(
        config: Config,
        num_workers=0,
        pin_memory=False,
        train_transform=None,
        val_transform=None,
):
    train_dataloader_set = []
    valid_dataloader_set = []
    for folder_name in os.listdir(config.data_dir):

        train_dataset = Stock_Dataset(data_dir=os.path.join(config.data_dir, folder_name), name=folder_name, predict_period=config.predict_period , day_windows=config.day_windows,
                                      flag="train", device=config.device)
        valid_dataset = Stock_Dataset(data_dir=os.path.join(config.data_dir, folder_name), name=folder_name,predict_period=config.predict_period ,day_windows=config.day_windows,
                                      flag="eval", device=config.device)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                      shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size,
                                      shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        train_dataloader_set.append(train_dataloader)
        valid_dataloader_set.append(valid_dataloader)


    return train_dataloader_set, valid_dataloader_set



def get_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建一个文件处理器，用于将日志记录到文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 创建一个控制台处理器，用于在控制台输出日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 添加处理器到Logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger









'''待拓展的功能'''