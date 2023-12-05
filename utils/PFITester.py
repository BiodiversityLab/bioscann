import argparse
import logging
import json
from pathlib import Path
import utils.TestDataloader as Dataloader
import random
import torch
import numpy as np

#this solves the issue Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# import utils.metrics as metrics


def read_json_file(file_path: Path) -> dict:
    """Read json file into dict."""
    log = logging.getLogger()

    with open(file_path) as json_file:
        channel_info = json.load(json_file)
        log.debug("Reading channel info data:")
        log.debug(f"{channel_info}")

        return channel_info


def get_channel_info(channel_info_file: Path) -> dict:

    channel_dict = read_json_file(channel_info_file)
    channels_per_api = {}
    for api in channel_dict.keys():
        channel_shape = channel_dict[api]
        num_channels = 0
        if len(channel_shape) <= 2:
            num_channels = 1
        else:
            num_channels = channel_shape[-1]
        channels_per_api[api] = num_channels

    return channels_per_api


def generate_index_mapping(dataset_length, seed=None):
    """Generate a random list with indexes that is of equal length to the dataloader's dataset."""

    def check_self_reference(index_list: list):
        """Return true if there is an element in the list has the same value as its index"""
        if len(index_list) <= 0:
            return True
        for index, value in enumerate(index_list):
            if index == value:
                return True
        return False

    # dataset_length = dataloader.test.__len__()
    index_list = [*range(0, dataset_length)]

    random_index_list = []
    if seed is None:
        # In order to prevent any element pointing to itself,
        # we randomize a list until we do not have an element in the list
        # that has the same value as index
        list_has_self_referencing_index = True
        while list_has_self_referencing_index:
            random_index_list = random.sample(index_list, len(index_list))
            list_has_self_referencing_index = check_self_reference(random_index_list)
    else:
        # If we have a seed, we only apply it and do not randomize further
        random_index_list = random.Random(seed).sample(index_list, len(index_list))

    return random_index_list


def num_channels_to_positions(ordered_dict):
    """Convert the number of channels for each key in the dict to the place of these channels
    in the channel stack"""
    current_channel_depth = 0
    positional_dict = {}
    for key in ordered_dict:
        num_channels = ordered_dict[key]
        position_list = [
            *range(current_channel_depth, current_channel_depth + num_channels)
        ]
        current_channel_depth += num_channels
        positional_dict[key] = position_list
    return positional_dict


def mix_features(input_images, feature_layers, index_mapping):
    """Shuffle all input images along a specific feature layer as specified by the index mapping."""
    shuffled_features_images = []
    # For all images, take the corresponding feature from the image with index
    # provided by the mapping
    for index, image in enumerate(input_images):
        wanted_index = index_mapping[index]
        feature_donor_image = input_images[wanted_index]
        new_image = np.copy(image)
        for layer in feature_layers:
            new_image[layer] = feature_donor_image[layer]
        shuffled_features_images.append(new_image)

    return np.array(shuffled_features_images)


def batch_predict(
    data_batch: torch.Tensor,
    ground_truth: torch.Tensor,
    model,
    mask: torch.Tensor,
    loss_fn,
):
    prediction = model(data_batch)
    masked_prediciton = prediction * mask
    loss = loss_fn(masked_prediciton, ground_truth)
    loss = loss.cpu().detach().numpy()
    predictions = prediction.cpu().detach().numpy()

    return predictions, loss


def run_PFI_test(
    dataset_folder: str,
    model,
    dataloader,
    device: torch.device,
    loss_fn,
    channel_info_file="",
    logging_level="INFO",
) -> None:
    """Run a Permutation Feature Importance test on the model over a dataset"""
    log = logging.getLogger()
    logging.basicConfig(level=logging_level)

    dataset_folder_path = Path(dataset_folder)

    # If we do not get a channel info file, look at the expected location
    if channel_info_file is None or channel_info_file == "":
        channel_info_file = dataset_folder_path / "channel_info.json"
    channel_info_file_path = Path(channel_info_file)
    assert (
        channel_info_file_path.exists()
    ), "No channel info file found in dataset folder"

    # Get the channel info from the file
    log.info("Finding channel information")
    channel_info = get_channel_info(channel_info_file_path)
    log.info(f"Channels per feature layer: \n{channel_info}")

    # Create result dict
    result_dict = {}
    metrics_keys = ["loss"]
    api_names = [*channel_info.keys()]
    api_names.append("benchmark")
    for api in api_names:
        result_dict[api] = {}
        for metric in metrics_keys:
            result_dict[api][metric] = []
    size_per_batch = []

    # Get the positions of each api in the channel stack
    position_dict = num_channels_to_positions(channel_info)
    log.debug(f'Position for each api in the channel stack:\n{position_dict}')

    # Permutate over the testing data
    for index, batch in enumerate(dataloader.dataloader):
        log.info(f"Running PFI on batch: {index+1}")
        # Unpack batch
        input_images = batch["image"].to(device)
        ground_truth = batch["output"].to(device)
        mask = batch["loss_mask"].to(device)

        # Skip if batch size is less than 2 as we can not permute
        if len(input_images) < 2:
            log.warning('Number of samples in current batch is less than 2, skipping')
            continue

        # Run model on original data to get a benchmark
        log.info("Benchmarking batch")
        bench_predictions, bench_loss = batch_predict(
            input_images, ground_truth, model, mask, loss_fn
        )
        result_dict["benchmark"]["loss"].append(bench_loss.tolist())
        size_per_batch.append(len(input_images))

        # Loop for each api
        for api in channel_info.keys():
            log.info(f"Running on feature: {api}")
            # Generate mapping betweeen data indexes to swap features
            index_map = generate_index_mapping(len(input_images))
            feature_layers_to_swap = position_dict[api]
            # Exchange channels in data to that of the randomized sample
            new_batch = mix_features(input_images.to(device).cpu().numpy(), feature_layers_to_swap, index_map)
            # Run model on new data
            predictions, loss = batch_predict(
                torch.from_numpy(new_batch).to(device), ground_truth, model, mask, loss_fn
            )
            result_dict[api]["loss"].append(loss.tolist())

    # Sum the metrics for all batches
    log.info("Summing metrics for all batches")
    for api in result_dict.keys():
        for index, metric_name in enumerate(result_dict[api]):
            metric_value = result_dict[api][metric_name]
            average_metric = calculate_average_over_batches(
                metric_value, size_per_batch
            )
            result_dict[api][metric_name] = average_metric

    # Calculate the performance difference for each metric
    pfi_dict = {}
    for api in result_dict.keys():
        if api != "benchmark":
            pfi_dict[api] = calculate_distance_to_benchmark(
                result_dict[api], result_dict["benchmark"]
            )

    return pfi_dict


def calculate_average_over_batches(value_list: list, batch_size_list: list) -> float:
    """Average the values by weighing them according to the size of the batch in relation to the
    entire dataset."""
    if value_list == [] or batch_size_list == []:
        log = logging.getLogger()
        log.warning(
            f"Can not calculate average of values {value_list}, with batch sizes {batch_size_list}"
        )
        return None

    batch_size_list = np.array(batch_size_list)
    total_samples = batch_size_list.sum()
    averaged_value = 0

    for index, value in enumerate(value_list):
        batch_weight = batch_size_list[index] / total_samples
        averaged_value += value * batch_weight

    return averaged_value


def calculate_distance_to_benchmark(pfi_dict: dict, benchmark_dict: dict) -> dict:
    """Calcualte the difference between the pfi and benchmark for each metric.
    A value more than one indicates that the benchmark scored higher,
    and a value less than the pfi scored higher when compared to the benchmark"""
    difference_dict = {}
    for metric_name in pfi_dict:
        pfi_value = pfi_dict[metric_name]
        benchmark_value = benchmark_dict[metric_name]
        difference_dict[metric_name] = benchmark_value - pfi_value
    return difference_dict


def load_test_data(
    dataset_folder_path: Path, image_size: list, batch_size: int
) -> Dataloader.PixelClassifierTestDataloader:
    """Load the specified data folder into a dataloader"""
    indata_folder = dataset_folder_path / "indata"
    outdata_folder = dataset_folder_path / "outdata"
    testdata_folder = dataset_folder_path / "testdata"
    dataloader = Dataloader.PixelClassifierTestDataloader(
        indata_folder, outdata_folder, testdata_folder, image_size, batch_size
    )
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel_info", action="store")
    parser.add_argument("--dataset", action="store", required=True)
    parser.add_argument("--logging", action="store", default="INFO")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--img_size", type=int, nargs="+", default=[256])
