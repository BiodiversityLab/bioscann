import numpy as np


def calculate_precision_recall_with_confidence(
    annotations: np.array, predictions: np.array
) -> tuple:
    """Calculate precision and recall of predictions with regard to the annotations"""
    # Get true positives by masking away all annotated negatives and summing the rest
    true_positives = (predictions * annotations).sum()

    # Invert the vector by setting all 1s to 0, and all 0s to 1
    inverse_annotations = annotations * (-1) + 1
    # Get false positives by masking with the inverse of the annotations and summing the rest
    false_positives = (predictions * inverse_annotations).sum()

    # Get false negatives by taking the difference between the predictions and 1,
    # while masking away all annotated negatives
    false_negatives = ((1 - predictions) * annotations).sum()

    # Calculate precision, if we have no predicted positives, set precision to 0
    if (true_positives + false_positives) == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    # Calculate recall, if we have no true positives or false negatives, set recall to 0
    if (true_positives + false_negatives) == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    return precision, recall


def precision_recall_test(
    annotations: np.array, predictions: np.array
) -> tuple:
    """Calculate precision and recall of predictions with regard to the annotations"""
    # Get true positives by masking away all annotated negatives and summing the rest
    true_positives = (predictions * annotations).sum()

    # Invert the vector by setting all 1s to 0, and all 0s to 1
    inverse_annotations = annotations * (-1) + 1
    # Get false positives by masking with the inverse of the annotations and summing the rest
    false_positives = (predictions * inverse_annotations).sum()

    # Get false negatives by taking the difference between the predictions and 1,
    # while masking away all annotated negatives
    false_negatives = ((1 - predictions) * annotations).sum()

    # Calculate precision, if we have no predicted positives, set precision to None
    if (true_positives + false_positives) == 0:
        precision = None
    else:
        precision = true_positives / (true_positives + false_positives)

    # Calculate recall, if we have no true positives or false negatives, set recall to None
    if (true_positives + false_negatives) == 0:
        recall = None
    else:
        recall = true_positives / (true_positives + false_negatives)

    return precision, recall


def calculate_positives_and_negatives(annotations: np.array, predictions: np.array) -> tuple:
    '''Calculate true positives, false positives, and false negatives of predictions 
    in regards to the annotations'''
    # Get true positives by masking away all annotated negatives and summing the rest
    true_positives = (predictions * annotations).sum()

    # Invert the vector by setting all 1s to 0, and all 0s to 1
    inverse_annotations = annotations * (-1) + 1
    # Get false positives by masking with the inverse of the annotations and summing the rest
    false_positives = (predictions * inverse_annotations).sum()

    # Get false negatives by taking the difference between the predictions and 1,
    # while masking away all annotated negatives
    false_negatives = ((1 - predictions) * annotations).sum()

    return true_positives, false_positives, false_negatives


def precision_recall_from_positives_and_negatives(
    true_positives: float, false_positives: float, false_negatives: float
) -> tuple:

    # Calculate precision, if we have no predicted positives, set precision to None
    if (true_positives + false_positives) == 0:
        precision = None
    else:
        precision = true_positives / (true_positives + false_positives)

    # Calculate recall, if we have no true positives or false negatives, set recall to None
    if (true_positives + false_negatives) == 0:
        recall = None
    else:
        recall = true_positives / (true_positives + false_negatives)

    return precision, recall


def calculate_binary_positives_and_negatives(annotations: np.array, predictions: np.array, mask: np.array) -> tuple:
    """Calculate binary true/false positives and false negatives."""
    rounded_predictions = np.round(predictions)

    true_positives = ((rounded_predictions == 1) & (annotations == 1)).sum()
    false_positives = ((rounded_predictions == 1) & (annotations == 0)).sum()
    false_negatives = ((rounded_predictions == 0) & (annotations == 1)).sum()

    if mask == None:
        false_pixels = abs(annotations - rounded_predictions).sum()
        total_pixels = len(annotations.flatten())
    else:
        target_pixels = np.where(mask > 0)
        false_pixels = abs(annotations[target_pixels]-rounded_predictions[target_pixels]).sum()
        total_pixels = len(annotations[target_pixels].flatten())

    correct_pixels = total_pixels-false_pixels

    return true_positives, false_positives, false_negatives, correct_pixels, total_pixels


def calculate_binary_precision_recall(
    annotations: np.array, predictions: np.array
) -> tuple:
    """Calculate binary precision by rounding the predicitons to 0 or 1 before calculation"""
    rounded_predictions = np.round(predictions)
    true_positives = ((rounded_predictions == 1) & (annotations == 1)).sum()
    false_positives = ((rounded_predictions == 1) & (annotations == 0)).sum()
    false_negatives = ((rounded_predictions == 0) & (annotations == 1)).sum()

    # Calculate precision, if we have no predicted positives, set precision to 0
    if (true_positives + false_positives) == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    # Calculate recall, if we have no true positives or false negatives, set recall to 0
    if (true_positives + false_negatives) == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    return precision, recall
