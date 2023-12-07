# from locale import normalize
import torch
import utils.AttentionPixelClassifier as attentionPixelClassifier
import utils.TestDataloader as testDataloader
import utils.PFITester as PFITester
import utils.feature_performance_test as fpt
import utils.metrics as metrics
from torch import nn
# import torch.optim as optim
import argparse
import os
import errno
import logging
import matplotlib.pyplot as plt
import numpy as np
# from tifffile import imread
from pathlib import Path
import mlflow

#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pdb


torch.cuda.empty_cache()


def ensure_dir(directory):
    """Ensure that the directory exists"""
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def mask_pred(pred, mask):
    res = pred * mask
    return res


def load_model(algorithm: str, input_channels, output_channels, device: torch.device, weights: str):
    """Load algorithm and model."""
    if str(algorithm).lower() == "attentionpixelclassifier":
        model = attentionPixelClassifier.AttentionPixelClassifier(
            input_numChannels=input_channels[0],
            output_numChannels=output_channels,
        ).to(device)
    elif str(algorithm).lower() == "attentionpixelclassifierlite":
        model = attentionPixelClassifier.AttentionPixelClassifierLite(
            input_numChannels=input_channels[0],
            output_numChannels=output_channels,
        ).to(device)
    elif str(algorithm).lower() == "attentionpixelclassifiermedium":
        model = attentionPixelClassifier.AttentionPixelClassifierMedium(
            input_numChannels=input_channels[0],
            output_numChannels=output_channels,
        ).to(device)
    elif str(algorithm).lower() == "attentionpixelclassifierlitedeep":
        model = attentionPixelClassifier.AttentionPixelClassifierLiteDeep(
            input_numChannels=input_channels[0],
            output_numChannels=output_channels,
        ).to(device)
    else:
        raise NotImplementedError(
            f"Algorithm {algorithm.lower()} is not supported"
        )

    if weights != "":
        print("load pretrained model")
        model.load_state_dict(torch.load(weights, map_location=device))
    return model


def main(opt: dict, init=True, model=''):
    """TODO: DOCUMENTATION"""

    # Setup logger for printing logs to terminal
    logging.basicConfig(level="INFO")
    log = logging.getLogger()

    # Set the device we will be using to train the model
    if opt.device != "cpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Create experiment folder
    experiment_path = os.path.join("train", opt.experiment_name)
    ensure_dir(experiment_path)

    # Load model
    if init:
        model = load_model(
            opt.algorithm,
            opt.input_channels,
            opt.output_channels,
            device,
            opt.weights
        )
        # Set model to eval mode
    model.eval()

    image_size = opt.img_size
    indata = os.path.join(opt.dataset, "indata")
    outdata = os.path.join(opt.dataset, "outdata")
    testdata = os.path.join(opt.dataset, "testdata")
    print(indata)
    print(outdata)
    print(testdata)

    mDataloader = testDataloader.PixelClassifierTestDataloader(
        indata, outdata, testdata, image_size, batch_size=opt.batch_size
    )
    lossFn = nn.BCELoss()

    # Run PFI test
    pfi_path = None
    if opt.pfi:
        log.info('Running Permutation Feature Importance test')
        #dataset_folder = Path(opt.dataset).parts[-2]
        dataset_folder = "/".join(Path(opt.dataset).parts[:-1]) 
        pfi_dict = permutation_feature_importance_test(
            Path(dataset_folder), model, mDataloader, device, lossFn
        )
        log.debug(f'PFI results: {pfi_dict}')
        # Save results to graph
        pfi_path = save_pfi_to_graph(pfi_dict, Path(experiment_path))
        log.debug(f'Saved PFI graph to {pfi_path}')

    # Calcuate binary precision and recall for the test features
    log.info('Running feature performance test')
    (
        feature_precision_dict,
        feature_recall_dict,
        feature_coverage_dict
    ) = fpt.calculate_feature_performance(model, mDataloader, device, lossFn)

    # Run regular model evaluation on data
    (
        loss,
        precision,
        recall,
        binary_precision,
        binary_recall,
        binary_accuracy
    ) = evaluate_model(model, mDataloader, lossFn, device)

    if opt.mlflow:

        # Compose log dict
        log_dict = {
            'loss': loss,
            'precision': precision,
            'recall': recall,
            'binary_accuracy': binary_accuracy,
            'binary precision': binary_precision,
            'binary recall': binary_recall,
        }
        artifacts_dict = {
            'PFI': pfi_path
        }
        nested_dict = {
            'precision on feature': feature_precision_dict,
            'recall on feature': feature_recall_dict,
            'coverage on feature': feature_coverage_dict
        }
        log_metrics_to_mlflow(
            log_dict,
            artifacts_dict,
            nested_dict
        )


def log_metrics_to_mlflow(
    metrics_dict: dict,
    artifacts_dict: dict,
    nested_dict: dict,
):
    '''Write metrics to mlflow experiment'''
    # Log metrics
    for metric in metrics_dict.keys():
        value = metrics_dict[metric]
        if value is not None:
            mlflow.log_metric(metric, value)

    # Log artifacts
    for artifact in artifacts_dict.keys():
        artifact_path = artifacts_dict[artifact]
        if artifact_path is not None and artifact_path.is_file:
            mlflow.log_artifact(artifact_path, artifact)

    # Log feature performance
    for metric in nested_dict.keys():
        internal_dict = nested_dict[metric]
        for feature in internal_dict.keys():
            value = internal_dict[feature]
            if value is not None:
                description = f'{metric} {feature}'
                mlflow.log_metric(description, value)
    return


def evaluate_model(
    model: attentionPixelClassifier,
    dataloader: testDataloader,
    loss_function: torch.nn.modules.loss,
    device: torch.device
) -> tuple:
    """Evaluate model on data provided by dataloader."""
    loss = 0.0
    true_positives = []
    false_positives = []
    false_negatives = []

    binary_true_positives = []
    binary_false_positives = []
    binary_false_negatives = []
    binary_correct_pixels = []
    binary_total_pixels = []

    for i, batch in enumerate(dataloader.dataloader):
        x, y, mask = (
            batch["image"],
            batch["output"],
            batch["loss_mask"],
        )
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pred = model(x)
        pred = mask_pred(pred, mask)

        batch_loss = loss_function(pred, y)
        batch_loss.backward()

        loss += batch_loss.item()

        predictions = pred.cpu().detach().numpy()
        annotations = y.cpu().detach().numpy()

        # Get true positives and true/false negatives to later calculate precision and recall
        tps, fps, fns, = metrics.calculate_positives_and_negatives(annotations, predictions)
        true_positives.append(tps)
        false_positives.append(fps)
        false_negatives.append(fns)
        # Get binary metrics
        b_tps, b_fps, b_fns, correct_pixels, total_pixels = metrics.calculate_binary_positives_and_negatives(
            annotations,
            predictions,
            mask
        )
        binary_true_positives.append(b_tps)
        binary_false_positives.append(b_fps)
        binary_false_negatives.append(b_fns)
        binary_correct_pixels.append(correct_pixels)
        binary_total_pixels.append(total_pixels)


    precision, recall = metrics.precision_recall_from_positives_and_negatives(
        np.array(true_positives).sum(),
        np.array(false_positives).sum(),
        np.array(false_negatives).sum()
    )

    binary_precision, binary_recall = metrics.precision_recall_from_positives_and_negatives(
        np.array(binary_true_positives).sum(),
        np.array(binary_false_positives).sum(),
        np.array(binary_false_negatives).sum(),
    )

    binary_accuracy = np.array(binary_correct_pixels).sum()/np.array(binary_total_pixels).sum()
    loss = loss / dataloader.dataloader.__len__()

    return loss, precision, recall, binary_precision, binary_recall, binary_accuracy


def permutation_feature_importance_test(
    dataset_folder, model, dataloader, device, loss_fn
):
    '''Run PFI test and return the loss difference for each feature'''
    # Check that the batch size is set to larger than 1, or else the PFI function won't work
    batch_size = dataloader.dataloader.batch_size
    if batch_size < 2:
        raise NotImplementedError(
            f'PFI tester requires a dataloader with a batch size >= 2, current size: {batch_size}'
            )

    # Run pfi test
    pfi_output = PFITester.run_PFI_test(
        dataset_folder, model, dataloader, device, loss_fn
    )

    # Extract the loss metric
    loss_dict = {}
    for feature in pfi_output.keys():
        feature_loss_diff = abs(pfi_output[feature]["loss"])
        loss_dict[feature] = feature_loss_diff
    return loss_dict


def save_pfi_to_graph(pfi_dict: dict, save_location: Path) -> Path:
    '''Save the pfi data to a matplot graph, returns a Path to the saved image'''
    ordered_dict = dict(sorted(pfi_dict.items(), key=lambda item: item[1], reverse=True))
    features = list(ordered_dict.keys())
    values = list(ordered_dict.values())
    # Create bar chart
    bar_width = 0.5
    plt.bar(range(len(ordered_dict)), values, tick_label=features, width=bar_width)
    # Add values for each feature as text in the bar
    for index, value in enumerate(values):
        rounded_value = round(value, ndigits=6)
        x_position = index - (bar_width/2)
        y_position = value
        plt.text(x_position, y_position, str(rounded_value))
    # Rotate and align text to make the x labels not overlap
    plt.xticks(rotation=30, ha='right')
    # Add a title
    plt.title('Feature Importance')
    # Add a y label
    plt.ylabel('Effect on loss')
    # Change layout to accomodate the room needed for the text
    plt.tight_layout()

    # Save plot as image
    figure_path = save_location / 'pfi_graph.png'
    plt.savefig(figure_path, bbox_inches='tight')
    return figure_path


def plot_pred(pred, mask, y, store_path, epoch, image_name):

    mask = mask[0].cpu().detach().numpy().transpose((1, 2, 0))
    prediction = pred[0].cpu().detach().numpy()
    prediction = prediction.transpose((1, 2, 0))
    prediction = np.clip(prediction, 0, 1)
    prediction *= 255
    prediction = np.array(
        [prediction[:, :, 0], prediction[:, :, 0], prediction[:, :, 0]]
    )
    prediction = prediction.transpose((1, 2, 0))

    res = y[0].cpu().detach().numpy()
    res = res.transpose(1, 2, 0)
    res = np.clip(res, 0, 1)
    res *= 255
    res = np.array([res[:, :, 0], res[:, :, 0], mask[:, :, 0] * 255])
    res = res.transpose((1, 2, 0))

    # con = np.concatenate((prediction, res), axis=1)
    image_name = image_name.replace("\\", "/")
    image_name = image_name.split("/")[-1].split(".")[0]
    image_path = "{}/epoch_{}_img_{}_pred_res.png".format(store_path, epoch, image_name)
    print("store: {}".format(image_path))
    # cv2.imwrite(image_path,con)
    return image_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--img_size", type=int, nargs="+", default=[256])
    parser.add_argument("--dataset", action="store", default="test_dataset")
    parser.add_argument(
        "--input_channels", type=int, nargs="+", action="store", default=[3, 3]
    )
    parser.add_argument("--output_channels", type=int, action="store", default=1)
    parser.add_argument("--experiment_name", action="store", default="test")
    parser.add_argument("--weights", action="store", default="")
    parser.add_argument("--device", action="store", type=str, default="0")
    parser.add_argument(
        "--algorithm", action="store", type=str, default="attentionpixelclassifier"
    )  # AttentionPixelclassifier
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument('--pfi', action='store_true')

    opt = parser.parse_args()

    if opt.mlflow:
        # mlflow.set_tracking_uri("file:/mnt/mlflow_tracking/mlruns")
        mlflow.set_tracking_uri('mlruns')
        mlflow.set_experiment(opt.experiment_name)

        mlflow.start_run()

        arguments = {}
        for arg in opt.__dict__:
            if opt.__dict__[arg] is not None:
                arguments[arg] = opt.__dict__[arg]
        mlflow.log_params(arguments)

    main(opt)

    if opt.mlflow:
        # mlflow.end_run(mlflow=mlflow)
        mlflow.end_run()
