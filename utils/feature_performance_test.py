import utils.metrics as metrics
import utils.AttentionPixelClassifier as attentionPixelClassifier
import utils.TestDataloader as testDataloader
import torch
import numpy as np
from tifffile import imread
import os

def predict(
    input_data,
    annotations,
    mask,
    model: attentionPixelClassifier,
    loss_function: torch.nn.modules.loss
) -> tuple:
    pred = model(input_data)
    pred = pred * mask

    batch_loss = loss_function(pred, annotations)
    batch_loss.backward()

    loss = batch_loss.item()

    predictions = pred.cpu().detach().numpy()
    annotations = annotations.cpu().detach().numpy()
    return predictions, annotations, loss


def calculate_feature_performance(
    model,
    dataloader: testDataloader,
    device: torch.device,
    loss_function: torch.nn.modules.loss
) -> tuple:
    def calc_pres_rec_on_features(feature_metrics_dict: dict) -> tuple:
        """Calculate precision and recall for each feature in our feature test"""
        precision_dict = dict((el, 0.0) for el in test_feature_names)
        recall_dict = dict((el, 0.0) for el in test_feature_names)
        for feature_name in feature_metrics_dict.keys():
            metrics_dict = feature_metrics_dict[feature_name]
            feature_true_positives = metrics_dict['true_positives']
            feature_false_positives = metrics_dict['false_positives']
            feature_false_negatives = metrics_dict['false_negatives']
            (
                feature_precision,
                feature_recall
            ) = metrics.precision_recall_from_positives_and_negatives(
                np.array(feature_true_positives),
                np.array(feature_false_positives),
                np.array(feature_false_negatives)
            )
            precision_dict[feature_name] = feature_precision
            recall_dict[feature_name] = feature_recall
        return precision_dict, recall_dict

    test_feature_names = dataloader.test.test_feature_names
    metric_names_list = ['true_positives', 'false_positives', 'false_negatives']
    feature_metrics_dict = dict((el, None) for el in test_feature_names)
    coverage_metrics_dict = dict((el, None) for el in test_feature_names)
    
    for feature in feature_metrics_dict:
        feature_metrics_dict[feature] = dict((metric, 0) for metric in metric_names_list)
        coverage_metrics_dict[feature] = dict((metric, 0) for metric in metric_names_list)

    for batch in dataloader.dataloader:
        # Run prediction on batch
        x, y, mask, test_feature_images_names = (
            batch["image"],
            batch["output"],
            batch["loss_mask"],
            batch["test_feature_images"],
        )
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        masks = mask.cpu().detach().numpy()
        batch_predictions, batch_annotations, _ = predict(x, y, mask, model, loss_function)

        # Calculate performance for each feature
        feature_samples_dict = dict((el, []) for el in test_feature_names)

        for test_feature_images_names_samples in test_feature_images_names:
            for feature in test_feature_names:
                feature_samples = [feature_sample for feature_sample in test_feature_images_names_samples.split(" ") if feature+'.' in feature_sample]
                feature_samples_dict[feature].extend(feature_samples)

        for feature in test_feature_names:
            
            samples = np.array([imread(image) for image in feature_samples_dict[feature] if os.path.exists(image)])

            number_of_images = batch_annotations.shape[0] if samples.shape[0] > batch_annotations.shape[0] else samples.shape[0]
            batch_predictions = batch_predictions[:number_of_images,:,:,:]

            samples = samples[:number_of_images,:,:]
            batch_annotations = batch_annotations[:number_of_images,:,:,:]
            masks = masks[:number_of_images,:,:,:]
            prediction_on_feature = batch_predictions * samples
            annotation_on_feature = batch_annotations * samples

            (
                feature_tps,
                feature_fps,
                feature_fns,
                feature_correct_pixels,
                feature_total_pixels
            ) = metrics.calculate_binary_positives_and_negatives(
                annotation_on_feature,
                prediction_on_feature,
                None
            )
            feature_metrics = {
                'true_positives': feature_tps,
                'false_positives': feature_fps,
                'false_negatives': feature_fns
            }
            for metric in feature_metrics.keys():
                feature_metrics_dict[feature][metric] += feature_metrics[metric]

            (
                coverage_tps,
                coverage_fps,
                coverage_fns,
                coverage_correct_pixels,
                coverage_total_pixels
            ) = metrics.calculate_binary_positives_and_negatives(
                masks,
                samples,
                None
            )
            coverage_metrics = {
                'true_positives': coverage_tps,
                'false_positives': coverage_fps,
                'false_negatives': coverage_fns
            }

            for metric in feature_metrics.keys():
                coverage_metrics_dict[feature][metric] += coverage_metrics[metric]
    # Calculate precision and recall for each feature
    feature_precision_dict, feature_recall_dict = calc_pres_rec_on_features(feature_metrics_dict)

    # Calculate coverage for each feature
    _, feature_coverage_dict = calc_pres_rec_on_features(coverage_metrics_dict)

    return feature_precision_dict, feature_recall_dict, feature_coverage_dict
