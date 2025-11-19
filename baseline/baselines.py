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
from LSTM import LSTM_Predictor


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parameter_sizes(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: nn.Module
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def convert_to_gpu(*data, device: str):
    """
    convert data from cpu to gpu, accelerate the running speed
    :param data: can be any type, including Tensor, Module, ...
    :param device: str
    """
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def create_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0):
    """
    create optimizer
    :param model: nn.Module
    :param optimizer_name: str, optimizer name
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay
    :return:
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer


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
    parser.add_argument('--num_features', type=int, default=7, help='features used in baseline')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    assert args.dataset_name in ['aminer', 'yelp'], f'Wrong value for dataset_name {args.dataset_name}!'
    return args


def get_popularity_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the popularity prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    rmsle = np.around(np.sqrt(mean_squared_error(labels, predicts)), 4)

    msle = np.around(mean_squared_error(labels, predicts), 4)

    pred_mean, label_mean = np.mean(predicts, axis=0), np.mean(labels, axis=0)
    pre_std, label_std = np.std(predicts, axis=0), np.std(labels, axis=0)
    pcc = np.around(np.mean((predicts - pred_mean) * (labels - label_mean) / (pre_std * label_std), axis=0), 4)

    male = np.around(mean_absolute_error(labels, predicts), 4)

    label_p2 = np.power(2, labels)
    pred_p2 = np.power(2, predicts)
    result = np.mean(np.abs(np.log2(pred_p2 + 1) - np.log2(label_p2 + 1)) / np.log2(label_p2 + 2))
    mape = np.around(result, 4)

    return {'rmsle': rmsle, 'msle': msle, 'pcc': pcc, 'male': male, 'mape': mape}


def evaluate_model_popularity_prediction(model_name: str, model: nn.Module,
                                         evaluate_idx_data_loader: DataLoader,
                                         evaluate_data: Data, loss_func: nn.Module):
    model.eval()
    with torch.no_grad():
        # store evaluate losses, trues and predicts
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120, disable=True)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_features, batch_labels = evaluate_data.features[evaluate_data_indices], evaluate_data.labels[
                evaluate_data_indices]
            batch_features = torch.from_numpy(batch_features).float().to(device)

            if model_name == 'MLP':
                predicts = model(batch_features).squeeze(dim=-1)
            elif model_name == 'LSTM':
                predicts = model(batch_features).squeeze(dim=-1)
            else:
                raise ValueError(f"Wrong value for model_name {args.model_name}!")
            labels = torch.from_numpy(batch_labels).float().to(predicts.device)
            labels = torch.log2(labels)
            loss = loss_func(input=predicts, target=labels)
            evaluate_total_loss += loss.item()
            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(predicts)
            evaluate_idx_data_loader_tqdm.set_description(
                f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= batch_idx + 1
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        # print(f'y_trues: {evaluate_y_trues}')
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)
        # print(f'y_predicts: {evaluate_y_predicts}')
        evaluate_metrics = get_popularity_prediction_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_popularity_prediction_args()
    device = args.gpu

    # get data for training, validation and testing
    train_data, val_data, test_data = get_popularity_prediction_data(dataset_name=args.dataset_name,
                                                                     time_interval=args.dataset_time_gap,
                                                                     model_name=args.model_name,
                                                                     num_features=args.num_features)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.labels))),
                                                batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.labels))),
                                              batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.labels))),
                                               batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []
    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        # args.load_model_name = f'{args.model_name}_seed{args.seed}'
        args.save_model_name = f'popularity_prediction_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name == 'MLP':
            dynamic_backbone = MLP_Predictor(input_dim=args.num_features)
        elif args.model_name == 'LSTM':
            dynamic_backbone = LSTM_Predictor(input_dim=args.num_features)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        model = dynamic_backbone
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')
        # follow previous work, we freeze the dynamic_backbone and only optimize the node_classifier
        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate,
                                     weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)
        save_model_folder = f"./saved_models/{args.model_name}_{str(time.time())}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.MSELoss()

        for epoch in range(args.num_epochs):

            model.train()

            # store train losses, trues and predicts
            train_total_loss, train_y_trues, train_y_predicts = 0.0, [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120, disable=True)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_features, batch_labels = train_data.features[train_data_indices], train_data.labels[
                    train_data_indices]
                batch_features = torch.from_numpy(batch_features).float().to(device)

                if args.model_name == 'MLP':
                    predicts = model(batch_features).squeeze(dim=-1)
                elif args.model_name == 'LSTM':
                    predicts = model(batch_features).squeeze(dim=-1)
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")

                labels = torch.from_numpy(batch_labels).float().to(predicts.device)
                labels = torch.log2(labels)
                loss = loss_func(input=predicts, target=labels)

                train_total_loss += loss.item()
                train_y_trues.append(labels)
                train_y_predicts.append(predicts)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_idx_data_loader_tqdm.set_description(
                    f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

            train_total_loss /= (batch_idx + 1)
            train_y_trues = torch.cat(train_y_trues, dim=0)
            train_y_predicts = torch.cat(train_y_predicts, dim=0)

            train_metrics = get_popularity_prediction_metrics(predicts=train_y_predicts, labels=train_y_trues)

            val_total_loss, val_metrics = evaluate_model_popularity_prediction(model_name=args.model_name,
                                                                               model=model,
                                                                               evaluate_idx_data_loader=val_idx_data_loader,
                                                                               evaluate_data=val_data,
                                                                               loss_func=loss_func)

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss MSLE: {train_total_loss:.4f}')
            for metric_name in train_metrics.keys():
                logger.info(f'train {metric_name}, {train_metrics[metric_name]:.4f}')
            logger.info(f'validate loss: {val_total_loss:.4f}')
            for metric_name in val_metrics.keys():
                logger.info(f'validate {metric_name}, {val_metrics[metric_name]:.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_total_loss, test_metrics = evaluate_model_popularity_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     evaluate_idx_data_loader=test_idx_data_loader,
                                                                                     evaluate_data=test_data,
                                                                                     loss_func=loss_func)
                logger.info(f'test loss: {test_total_loss:.4f}')
                for metric_name in test_metrics.keys():
                    logger.info(f'test {metric_name}, {test_metrics[metric_name]:.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics.keys():
                higher_better = False
                if metric_name == 'rmsle':
                    val_metric_indicator.append((metric_name, val_metrics[metric_name], higher_better))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')
        test_total_loss, test_metrics = evaluate_model_popularity_prediction(model_name=args.model_name,
                                                                             model=model,
                                                                             evaluate_idx_data_loader=test_idx_data_loader,
                                                                             evaluate_data=test_data,
                                                                             loss_func=loss_func)

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        logger.info(f'test loss: {test_total_loss:.4f}')
        for metric_name in test_metrics.keys():
            test_metric = test_metrics[metric_name]
            logger.info(f'test {metric_name}, {test_metric:.4f}')
            test_metric_dict[metric_name] = test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in
                                 val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in
                             test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in val_metric_all_runs[0].keys():
        logger.info(
            f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
        logger.info(
            f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
            f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(
            f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(
            f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
            f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
