from sklearn.metrics import mean_absolute_error, accuracy_score, mean_absolute_percentage_error
import torch
import random
from collections import Counter
import numpy as np
import argparse
import faiss
from datasets import load_dataset



# Data names for the DREAM dataset
CONCEPTS_101_DATA_NAMES = ['wearable_sunglasses1', 'wearable_sunglasses2', 'plushie_panda', 'things_cup2', 'plushie_tortoise',
                    'person_2', 'plushie_pink', 'person_1', 'pet_cat7', 'person_3', 'jewelry_earring',
                    'plushie_teddybear', 'toy_pikachu1', 'things_book2', 'plushie_penguin', 'scene_barn',
                    'pet_cat5', 'transport_car3', 'actionfigure_2', 'plushie_dice', 'luggage_purse4',
                    'things_headphone2', 'plushie_bunny', 'transport_tank', 'instrument_music1', 'pet_dog4',
                    'decoritems_houseplant2', 'decoritems_houseplant1', 'pet_dog1', 'flower_1',
                    'actionfigure_3', 'things_headphone1', 'transport_car2', 'plushie_happysad',
                    'scene_waterfall', 'things_corkscrew', 'flower_2', 'transport_car1', 'wearable_shoes2',
                    'things_bottle1', 'plushie_lobster', 'instrument_music3', 'transport_motorbike1',
                    'things_cup3', 'decoritems_houseplant3', 'transport_car6', 'instrument_music2',
                    'scene_garden', 'plushie_cow', 'toy_unicorn']

def parser_args():
    parser = argparse.ArgumentParser(description="A script for training and evaluating DSiRe")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.", default=42)
    parser.add_argument("--rank", type=int, help="Rank of LoRA.", default=32, choices=[8, 16, 32])
    parser.add_argument("--cache_dir", type=str, default="./.cache/lora_wise",
                        help="path to cache the dataset, prevents downloading the entire dataset for every fine-tuned model in distributed mode")
    parser.add_argument("--subset", type=str,
                        choices=["high_32", "medium_32", "medium_32", "medium_16", "low_32", "low_16", "low_8"],
                        default='medium_32',
                        help='LoRA WiSE dataset subset, options are "high_32", "medium_32", "medium_32", '
                             '"medium_16", "low_32", "low_16", "low_8" for more details https://huggingface.co/datasets/MoSalama98/LoRA-WiSE',
                        required=True)
    parser.add_argument("--dataset", type=str, default="MoSalama98/LoRA-WISE",
                        help="dataset path, supports hugging face datasets")

    args = parser.parse_args()
    return args

def prepare_faiss(data_names_train, num_svd, all_data_dict, data_sizes):
    """
    Prepare FAISS indexes for training data.

    Parameters:
    - data_names_train (list): List of training data names.
    - num_svd (int): Number of singular values to use.
    - all_data_dict (dict): Dictionary containing all data.
    - data_sizes (list): List of data sizes.

    Returns:
    - indexes (dict): Dictionary of FAISS indexes.
    - label_mapping (dict): Dictionary of label mappings.
    """
    indexes = {}
    label_mapping = {}
    for layer in all_data_dict[data_names_train[0]][data_sizes[0]].keys():
        # Preparing data and labels
        layer_data = []
        y_layer_data = []
        for data_name in data_names_train:
            for size in data_sizes:
                data = torch.stack([all_data_dict[data_name][size][layer][:num_svd]]).numpy()
                layer_data.extend(data)
                y_layer_data.extend([size] * len(data))

        # Convert list to numpy array for Faiss
        layer_data = np.vstack(layer_data).astype(np.float32)

        # Create the index
        index = faiss.IndexFlatL2(layer_data.shape[1])

        # Add data to index
        index.add(layer_data, 1)

        # Store index and labels
        indexes[layer] = index
        label_mapping[layer] = np.array(y_layer_data)

    return indexes, label_mapping

def classify_faiss(data_points, indexes, label_mapping, num_svd):
    """
    Classify data points using FAISS indexes.

    Parameters:
    - data_points (list): List of data points to classify.
    - indexes (dict): Dictionary of FAISS indexes.
    - label_mapping (dict): Dictionary of label mappings.
    - num_svd (int): Number of singular values to use.

    Returns:
    - pred_key (dict): Dictionary of predictions.
    """
    pred_key = {key: [] for key in indexes.keys()}
    for svd_data_point in data_points:
        for layer, size_dict in indexes.items():
            # Query the index
            D, I = indexes[layer].search(np.array(svd_data_point[layer][:num_svd]).reshape(1, -1).astype(np.float32), 1)
            # Retrieve labels for the nearest neighbor
            pred = label_mapping[layer][I[0]]
            pred_key[layer].extend(pred.tolist())
    return pred_key

def prepare_val_data(all_data_dict, data_names_val, data_sizes):
    """
    Prepare validation data.

    Parameters:
    - all_data_dict (dict): Dictionary containing all data.
    - data_names_val (list): List of validation data names.
    - data_sizes (list): List of data sizes.

    Returns:
    - data_points (list): List of validation data points.
    - y_true (list): List of true labels.
    """
    y_true = []
    data_points = []
    for data_size in data_sizes:
        for data_name in data_names_val:
            data_points.append(all_data_dict[data_name][data_size])
            y_true.append(data_size)
    return data_points, y_true

def eval(y_true, y_pred_val):
    """
    Evaluate the model predictions.

    Parameters:
    - y_true (list): List of true labels.
    - y_pred_val (list): List of predicted labels.

    Returns:
    - accuracy (float): Accuracy of the model.
    - mse (float): Mean absolute error of the model.
    - mape (float): Mean absolute percentage error of the model.
    """
    accuracy = accuracy_score(y_true, y_pred_val)
    mse = mean_absolute_error(y_true, y_pred_val)
    mape = mean_absolute_percentage_error(y_true, y_pred_val)
    print(f'Val Accuracy: {accuracy * 100:.2f}%')
    print(f'Val MAPE: {mape * 100:.2f}%')
    print("Mean Absolute Error (MAE):", mse)
    return accuracy * 100, mse, mape * 100

def normalize_data(data_dict, min_val, max_val):
    """
    Normalize data.

    Parameters:
    - data_dict (dict): Dictionary containing data.
    - min_val (float): Minimum value for normalization.
    - max_val (float): Maximum value for normalization.

    Returns:
    - data_dict (dict): Normalized data dictionary.
    """
    for data_name in data_dict.keys():
        for data_size in data_dict[data_name].keys():
            for key in data_dict[data_name][data_size].keys():
                data_point = data_dict[data_name][data_size][key]
                data_dict[data_name][data_size][key] = (data_point - min_val) / (max_val - min_val)
    return data_dict

def get_min_max(data_dict, data_names_train):
    """
    Get minimum and maximum values from the data.

    Parameters:
    - data_dict (dict): Dictionary containing data.
    - data_names_train (list): List of training data names.

    Returns:
    - min_val (float): Minimum value.
    - max_val (float): Maximum value.
    """
    min_val, max_val = 1000, 0
    for data_name in data_names_train:
        for data_size in data_dict[data_name].keys():
            for key in data_dict[data_name][data_size].keys():
                data_point = data_dict[data_name][data_size][key]
                if min_val > data_point.min():
                    min_val = data_point.min()
                if max_val < data_point.max():
                    max_val = data_point.max()
    return min_val, max_val

def svd(lora_weights, is_ba=False, current_lora_rank=32):
    """
    Perform SVD on LoRA weights.

    Parameters:
    - lora_weights (list): List of LoRA weight dictionaries.
    - is_ba (bool): Flag to indicate if B*A transformation is used.
    - current_lora_rank (int): Rank for SVD.

    Returns:
    - svds (list): List of SVD results.
    """
    svds = []
    for lora in lora_weights:
        svd_results = {}
        for key, tensor in lora.items():
            tensor = tensor.to(torch.float32)
            if torch.cuda.is_available():
                tensor = tensor.to('cuda')
            if is_ba:
                U, S, V = torch.svd_lowrank(tensor, q=current_lora_rank)
            else:
                U, S, V = torch.svd_lowrank(tensor, q=current_lora_rank)
            svd_results[key] = S.detach().cpu()  # Store the singular values
        svds.append(svd_results)
    return svds

def prepare_BA_matrix(lora_weights):
    """
    Prepare B*A transformations for LoRA weights.

    Parameters:
    - lora_weights (dict): Dictionary of LoRA weights.

    Returns:
    - lora_results (dict): Dictionary of B*A results.
    """
    lora_results = {}
    for key in lora_weights.keys():
        parts = key.split('.')
        lora_type = parts[-1]
        up_key = '.'.join(parts[:-1])
        if lora_type == 'down':
            continue  # Skip A here as we need its pair B to perform multiplication
        key_A = f"{up_key}.down"
        key_B = f"{up_key}.up"
        if key_A in lora_weights and key_B in lora_weights:
            if torch.cuda.is_available():
                B = torch.tensor(lora_weights[key_B]).to('cuda').unsqueeze(dim=1)
                A = torch.tensor(lora_weights[key_A]).to('cuda').unsqueeze(dim=0)
            lora_results[up_key] = (B @ A).cpu()
    return lora_results

def preprocess_weights_dict(lora_weights):
    """
    Preprocess weights dictionary to simplify keys.

    Parameters:
    - lora_weights (dict): Dictionary of LoRA weights.

    Returns:
    - simplified_weights (dict): Dictionary with simplified keys.
    """
    lora_weights = {key[::-1].replace('.0.', '.', 1)[::-1].replace('.lora.', '.') if 'to_out' in key
     else key.replace('.lora.', '.'): torch.tensor(value, dtype=torch.float32) for key, value in
     lora_weights.items()}

    simplified_weights = {}
    for key in lora_weights:
        parts = key.split('.')
        new_key = '.'.join(parts[:-1])
        simplified_weights[new_key] = lora_weights[key]
    return simplified_weights


def load_weights(data_sizes, data_name, dataset):
    """

    Parameters:
    - output_dir (str): Directory containing model checkpoints.
    - data_sizes (list): List of data sizes.
    - data_name (str): Name of the data.
    - dataset

    Returns:
    - dic_weights_ds (dict): Dictionary of LoRA weights.
    - dic_weights_ds_ba (dict): Dictionary of B*A transformed LoRA weights.
    """

    def filter_data(example):
        return [name == data_name for name in example]

    dic_weights_ds = {}
    dic_weights_ds_ba = {}
    samples_data = dataset.filter(filter_data, input_columns='name', batched=True).to_pandas()
    for data_size in data_sizes:
        try:
            lora_weights = samples_data[(samples_data['label'] == data_size)].drop(columns=['label', 'name']).squeeze().to_dict()
            lora_weights = preprocess_weights_dict(lora_weights)
            lora_weights_ba = prepare_BA_matrix(lora_weights)
            dic_weights_ds[data_size] = lora_weights
            dic_weights_ds_ba[data_size] = lora_weights_ba
        except Exception as e:
            print(f"Error loading weights for {data_name}: {e}")
    del samples_data
    return dic_weights_ds, dic_weights_ds_ba

def extract_data(dataset, rank, data_names, data_sizes):
    """
    Extract SVD data from LoRA weights.

    Parameters:
    - dataset: the loaded dataset.
    - rank (int): Rank for SVD.
    - data_names (list): List of data names.
    - data_sizes (list): List of data sizes.

    Returns:
    - all_data_dict (dict): Dictionary containing all extracted data.
    """
    all_data_dict = {}
    for data_name in data_names:
        dict_weights, dict_weights_ba = load_weights(data_sizes, data_name, dataset)
        svd_dict_results = {data_size: svd([data_size_loras], current_lora_rank=rank) for data_size, data_size_loras in dict_weights.items()}
        del dict_weights
        svd_dict_results_ba = {data_size: svd([data_size_loras], is_ba=True, current_lora_rank=rank) for data_size, data_size_loras in dict_weights_ba.items()}
        del dict_weights_ba
        concatenated_dict = {}
        for data_size in svd_dict_results.keys():
            concatenated_dict[data_size] = {**svd_dict_results[data_size][0], **svd_dict_results_ba[data_size][0]}
        all_data_dict[data_name] = concatenated_dict
        del concatenated_dict
    return all_data_dict

def return_data(args):
    """
    Return data sizes and names based on the dataset type.

    Parameters:
    - args (argparse.Namespace): Arguments from the command line.

    Returns:
    - data_sizes (list): List of data sizes.
    - data_names (list): List of data names.
    """
    if  "medium" in args.subset:
        data_sizes = [1, 10, 20, 30, 40, 50]
        data_names = list(range(1, 51))
    elif "high" in args.subset:
        data_sizes = [1, 10, 100, 500, 1000]
        data_names = list(range(1, 51))
    else:
        data_sizes = list(range(1, 7))
        data_names = CONCEPTS_101_DATA_NAMES
    return data_sizes, data_names

def main(args):
    """
    Main function to run the script.

    Parameters:
    - args (argparse.Namespace): Arguments from the command line.
    """
    dataset = load_dataset(args.dataset, name=args.subset)['train']
    data_sizes, data_names = return_data(args)
    all_data_dict = extract_data(dataset, args.rank, data_names, data_sizes)
    del dataset
    random.seed(args.seed)
    train_size = 15
    for experiment in range(10):
        print(f'experiment num: {experiment}')
        train_set_idx = random.sample(range(len(data_names)), train_size)
        data_names_train = [data_names[i] for i in train_set_idx]
        data_names_val = [data_name for data_name in data_names if data_name not in data_names_train]
        min_val, max_val = get_min_max(all_data_dict, data_names_train)
        all_data_dict_norm = normalize_data(all_data_dict, min_val, max_val)
        knns, label_mapping = prepare_faiss(data_names_train, args.rank, all_data_dict_norm, data_sizes)
        data_points, y_true = prepare_val_data(all_data_dict_norm, data_names_val, data_sizes)
        pred_layers_val = classify_faiss(data_points, knns, label_mapping, args.rank)
        # Model
        y_pred_val = [Counter(labels).most_common(1)[0][0] for labels in zip(*pred_layers_val.values())]
        eval(y_true, y_pred_val)

if __name__ == '__main__':
    args = parser_args()
    main(args)
