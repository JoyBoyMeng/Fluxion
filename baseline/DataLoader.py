from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels

def get_popularity_prediction_data(dataset_name: str, time_interval: str, model_name: str, num_features: int):
    # Load data and train val test split
    if dataset_name == 'aminer':
        train_data_feature = np.load('./processed_data/{}/features_train.npy'.format(dataset_name))
        val_data_feature = np.load('./processed_data/{}/features_val.npy'.format(dataset_name))
        test_data_feature = np.load('./processed_data/{}/features_test.npy'.format(dataset_name))
        train_data_label = np.load('./processed_data/{}/labels_train.npy'.format(dataset_name))
        val_data_label = np.load('./processed_data/{}/labels_val.npy'.format(dataset_name))
        test_data_label = np.load('./processed_data/{}/labels_test.npy'.format(dataset_name))
        if model_name == 'MLP':
            train_data_feature = train_data_feature[:, -1, :num_features]
            val_data_feature = val_data_feature[:, -1, :num_features]
            test_data_feature = test_data_feature[:, -1, :num_features]
        elif model_name == 'LSTM':
            train_data_feature = train_data_feature[:, :, :num_features]
            val_data_feature = val_data_feature[:, :, :num_features]
            test_data_feature = test_data_feature[:, :, :num_features]
        train_data = Data(features=train_data_feature, labels=train_data_label)
        val_data = Data(features=val_data_feature, labels=val_data_label)
        test_data = Data(features=test_data_feature, labels=test_data_label)
        # the setting of seed follows previous works
        random.seed(2025)

        return train_data, val_data, test_data
    else:
        print('To do')
