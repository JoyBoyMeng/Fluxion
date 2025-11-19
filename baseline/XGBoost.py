import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import argparse
import random
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from EarlyStopping import EarlyStopping
from torch.utils.data import DataLoader
from DataLoader import Data, get_popularity_prediction_data, get_idx_data_loader
from MLP import MLP_Predictor

def get_popularity_prediction_args():
    """
    get the args for the popularity prediction task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the popularity prediction task on MLP')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='aminer',
                        choices=['aminer', 'yelp'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--model_name', type=str, default='MLP', help='name of the model', choices=['MLP', 'LSTM'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'],
                        help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--dataset_time_gap', type=str, default='half_year',
                        help='time interval for popularity prediction')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    assert args.dataset_name in ['aminer', 'yelp'], f'Wrong value for dataset_name {args.dataset_name}!'
    return args

