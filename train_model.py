import argparse
import errno
import math
import os
import pdb
from locale import normalize

import cv2
import matplotlib
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch
import torch.optim as optim
from torch import nn

import utils.AttentionPixelClassifier as attentionPixelClassifier
import utils.Dataloader as dataloader
import test_model as test

torch.cuda.empty_cache()


def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def mask_pred(pred, mask):
    res = pred * mask
    return res


def main(opt):
    experiment_path = os.path.join(opt.workdir, "train", opt.experiment_name)
    print('Training results will be stored at', experiment_path)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    if opt.mlflow:
        run = mlflow.active_run()
        if run:
            current_run_info = run.info
            experiment_id = current_run_info.experiment_id
            client = MlflowClient()
            experiment = client.get_experiment(experiment_id)
            experiment_name = experiment.name
            artifact_location = experiment.artifact_location
        print('Mlflow artifact path:', artifact_location)

    # set the device we will be using to train the model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    # print(device)
    if opt.device != "cpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if str(opt.algorithm).lower() == "attentionpixelclassifier":
        model = attentionPixelClassifier.AttentionPixelClassifier(
            input_numChannels=opt.input_channels[0],
            output_numChannels=opt.output_channels,
        ).to(device)
        image_size = opt.img_size
    elif str(opt.algorithm).lower() == "attentionpixelclassifierlite":
        model = attentionPixelClassifier.AttentionPixelClassifierLite(
            input_numChannels=opt.input_channels[0],
            output_numChannels=opt.output_channels,
        ).to(device)
        image_size = opt.img_size
    elif str(opt.algorithm).lower() == "attentionpixelclassifiermedium":
        model = attentionPixelClassifier.AttentionPixelClassifierMedium(
            input_numChannels=opt.input_channels[0],
            output_numChannels=opt.output_channels,
        ).to(device)
        image_size = opt.img_size
    elif str(opt.algorithm).lower() == "attentionpixelclassifierlitedeep":
        model = attentionPixelClassifier.AttentionPixelClassifierLiteDeep(
            input_numChannels=opt.input_channels[0],
            output_numChannels=opt.output_channels,
        ).to(device)
        image_size = opt.img_size
    elif str(opt.algorithm).lower() == "attentionpixelclassifierflex":
        n_channels_per_layer = opt.n_channels_per_layer
        n_channels_per_layer = np.array(n_channels_per_layer.split(',')).astype(int)
        if opt.n_coefficients_per_upsampling_layer != None:
            n_coefficients_per_upsampling_layer = opt.n_coefficients_per_upsampling_layer
            n_coefficients_per_upsampling_layer = np.array(n_coefficients_per_upsampling_layer.split(',')).astype(int)
        else:
            n_coefficients_per_upsampling_layer = opt.n_coefficients_per_upsampling_layer
        model = attentionPixelClassifier.AttentionPixelClassifierFlex(
            input_numChannels=opt.input_channels[0],
            output_numChannels=opt.output_channels,
            n_channels_per_layer=n_channels_per_layer,
            n_coefficients_per_upsampling_layer=n_coefficients_per_upsampling_layer
        ).to(device)
        image_size = opt.img_size

    if opt.weights != "":
        print("load pretrained model")
        model.load_state_dict(torch.load(opt.weights))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of trainable parameters: {}".format(params))

    indata = os.path.join(opt.dataset, "indata")
    outdata = os.path.join(opt.dataset, "outdata")
    print(indata)
    print(outdata)
    if opt.validation != "":
        val_indata = os.path.join(opt.validation, "indata")
        val_outdata = os.path.join(opt.validation, "outdata")
        mDataloader = dataloader.PixelClassifierDataloader(
            indata,
            outdata,
            image_size,
            cutmix=False,
            batch_size=opt.batch_size,
            val_indata=val_indata,
            val_outdata=val_outdata,
            loss_mask= not opt.no_loss_mask,
        )
    else:
        mDataloader = dataloader.PixelClassifierDataloader(
            indata,
            outdata,
            image_size,
            cutmix=False,
            batch_size=opt.batch_size,
            loss_mask= not opt.no_loss_mask,
        )

    lossFn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999))

    best_score = 0
    # pdb.set_trace()
    for epoch in range(opt.epochs):
        # torch.enable_grad()
        print("epoch: {}".format(epoch))
        train_loss = 0
        for i, batch in enumerate(mDataloader.train_dataloader):
            if not opt.no_loss_mask:
                x, y, mask, image_names = (
                    batch["image"],
                    batch["output"],
                    batch["loss_mask"],
                    batch["image_name"],
                )
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optimizer.zero_grad()
                pred = model(x)
                pred = mask_pred(pred, mask)
            else:
                x, y, image_names = (
                    batch["image"],
                    batch["output"],
                    batch["image_name"],
                )
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)

            # number_of_pixels_in_batch = opt.img_size*opt.img_size * y.sum()
            # zero_values = number_of_pixels_in_batch - y.sum()
            # if not opt.no_loss_mask:
            #     target_pixels = np.where(mask > 0)
            #     loss = lossFn(pred[target_pixels], y[target_pixels])
            # else:
            loss = lossFn(pred, y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        val_loss = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_binary_precision = 0.0
        val_binary_recall = 0.0
        val_binary_accuracy = 0.0

        # torch.no_grad()

        for i, batch in enumerate(mDataloader.val_dataloader):
            optimizer.zero_grad()
           
            if not opt.no_loss_mask:
                val_x, val_y, val_mask, val_image_names = (
                    batch["image"],
                    batch["output"],
                    batch["loss_mask"],
                    batch["image_name"],
                )
                val_x, val_y, val_mask = (
                    val_x.to(device),
                    val_y.to(device),
                    val_mask.to(device),
                )
                val_pred = model(val_x)
                val_pred = mask_pred(val_pred, val_mask)
            else:
                val_x, val_y, val_image_names = (
                    batch["image"],
                    batch["output"],
                    batch["image_name"],
                )
                val_mask = ''
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_pred = model(val_x)
            # if not opt.no_loss_mask:
            #     target_pixels = np.where(mask > 0)
            #     loss = lossFn(val_pred[target_pixels], val_y[target_pixels])
            # else:
            loss = lossFn(val_pred, val_y)
            # Calculate precision and recall
            # batch_val_precision, batch_val_recall = calculate_precision_recall(val_y, val_pred)
            # val_precision += batch_val_precision.cpu().detach().numpy()
            # val_recall += batch_val_recall.cpu().detach().numpy()
            predictions = val_pred.cpu().detach().numpy()
            annotations = val_y.cpu().detach().numpy()
            (
                batch_val_precision,
                batch_val_recall,
            ) = calculate_precision_recall_with_confidence(annotations, predictions)
            val_precision += batch_val_precision
            val_recall += batch_val_recall

            # Calculate binary precision and recall
            (
                batch_val_binary_precision,
                batch_val_binary_recall,
                binary_val_accuracy
            ) = calculate_binary_precision_recall(annotations, predictions, val_mask)
            val_binary_precision += batch_val_binary_precision
            val_binary_recall += batch_val_binary_recall
            val_binary_accuracy += binary_val_accuracy

            val_loss += loss.item()

            if opt.target_img_name == '':
                if epoch==0 and i == 0:
                    target_img_name = batch["image_name"][0]
            else:
                dirname = os.path.dirname(batch["image_name"][0])
                filename = os.path.basename(opt.target_img_name)
                target_img_name = os.path.join(dirname,filename)

            if epoch==0 and i == 0:
                print('Using instance', target_img_name, 'for validation plot')

            file_print_switch = sum([target_img_name in name for name in batch["image_name"]])
            if opt.plot:
                if file_print_switch==1:
                    val_img_index = np.where(np.array(batch["image_name"]) == target_img_name)[0].__int__()
                    output_val_image(val_pred,val_mask,val_x,val_y,experiment_path,epoch,val_image_names,val_img_index)

        current_score = (val_binary_accuracy/mDataloader.val_dataloader.__len__() +
                         val_binary_precision/mDataloader.val_dataloader.__len__() +
                         val_binary_recall/mDataloader.val_dataloader.__len__() -
                         val_loss/mDataloader.val_dataloader.__len__())
        if opt.mlflow:
            mlflow.log_metric(
                "train_loss", float(train_loss) / mDataloader.train_dataloader.__len__(), step=epoch
            )
            mlflow.log_metric(
                "val_loss", float(val_loss) / mDataloader.val_dataloader.__len__(), step=epoch
            )
            mlflow.log_metric(
                "val_precision",
                float(val_precision) / mDataloader.val_dataloader.__len__(), step=epoch
            )
            mlflow.log_metric(
                "val_recall", float(val_recall) / mDataloader.val_dataloader.__len__(), step=epoch
            )
            # Log binary precision and recall
            mlflow.log_metric(
                "val_binary_precision",
                float(val_binary_precision) / mDataloader.val_dataloader.__len__(), step=epoch
            )
            mlflow.log_metric(
                "val_binary_recall",
                float(val_binary_recall) / mDataloader.val_dataloader.__len__(), step=epoch
            )
            mlflow.log_metric(
                "val_binary_accuracy",
                float(val_binary_accuracy) / mDataloader.val_dataloader.__len__(), step=epoch
            )
            mlflow.log_metric(
                "val_score",
                float(current_score), step=epoch
            )

        print(
            "epoch: {}, train_loss: {}, val_loss: {}, val_binary precision: {}, val_binary recall: {}, val_binary accuracy: {}, val_score: {}".format(
                epoch,
                loss/mDataloader.val_dataloader.__len__(),
                val_loss/mDataloader.val_dataloader.__len__(),
                val_binary_precision/mDataloader.val_dataloader.__len__(),
                val_binary_recall/mDataloader.val_dataloader.__len__(),
                val_binary_accuracy/mDataloader.val_dataloader.__len__(),
                current_score
            )
        )
        if current_score > best_score:
            best_score = current_score
            torch.save(
                model.state_dict(), os.path.join(experiment_path, "best_model.pth")
            )
            if opt.mlflow:
                mlflow.log_artifact(
                    os.path.join(experiment_path, "best_model.pth"), "weights"
                )

        if epoch != 0 and epoch % 100 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    experiment_path, "epoch_{}_loss_{}.pth".format(epoch, loss)
                ),
            )
            if opt.mlflow:
                mlflow.log_artifact(
                    os.path.join(
                        experiment_path, "epoch_{}_loss_{}.pth".format(epoch, loss)
                    ),
                    "weights",
                )

    torch.save(model.state_dict(), os.path.join(experiment_path, "model.pth"))
    if opt.mlflow:
        mlflow.log_artifact(os.path.join(experiment_path, "model.pth"), "weights")
    
    if opt.test_dataset:
        model.load_state_dict(torch.load(os.path.join(experiment_path, "best_model.pth"),map_location=device))
        opt.dataset = opt.test_dataset
        test.main(opt, init=False, model=model)


def output_val_image(val_pred,val_mask,val_x,val_y,experiment_path,epoch,val_image_names,val_img_index):
    # image_path = plot_pred(pred, y,experiment_path,str(epoch), image_names[0])
    # plot val:
    image_path = plot_pred(
        val_pred,
        val_mask,
        val_y,
        experiment_path,
        str(epoch),
        val_image_names,
        val_img_index
    )
    #print(image_path)
    # if epoch == 0:
    #     images_paths = plot_training_images(val_x.cpu().detach().numpy(), experiment_path)
    #     if opt.mlflow:
    #         for train_batch_image_path in images_paths:
    #             mlflow.log_artifact(train_batch_image_path, "images")
    if opt.mlflow:
        mlflow.log_artifact(image_path, "images")


def calculate_precision_recall_with_confidence(
    annotations: np.array, predictions: np.array
) -> tuple:
    """Calculate precision and recall of predictions with regard to the annotations."""
    # Get true positives by masking away all annotated negatives and summing the rest
    true_positives = (predictions * annotations).sum()

    # Invert the vector by setting all 1s to 0, and all 0s to 1
    inverse_annotations = annotations * (-1) + 1
    # Get false positives by masking with the inverse of the annotations and summing the rest
    false_positives = (predictions * inverse_annotations).sum()

    # Get false negatives by taking the difference between the predictions and 1, while masking away all annotated negatives
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


def calculate_binary_precision_recall(annotations: np.array, predictions: np.array, mask:np.array) -> tuple:
    """Calculate binary precision by rounding the predicitons to 0 or 1 before calculation."""
    target_pixels = np.where(mask > 0)
    rounded_predictions = np.round(predictions)

    true_positives = ((rounded_predictions[target_pixels] == 1) & (annotations[target_pixels] == 1)).sum()
    false_positives = ((rounded_predictions[target_pixels] == 1) & (annotations[target_pixels] == 0)).sum()
    false_negatives = ((rounded_predictions[target_pixels] == 0) & (annotations[target_pixels] == 1)).sum()

    false_pixels = abs(annotations[target_pixels]-rounded_predictions[target_pixels]).sum()
    total_pixels = len(annotations[target_pixels])
    correct_pixels = total_pixels-false_pixels
    accuracy = correct_pixels/total_pixels

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
    return precision, recall, accuracy


def plot_training_images(images, path):
    #pdb.set_trace()
   # number_of_images = images.shape[0]
    images_paths = []
    for index, image in enumerate(images):
        if index >= 4:
            break
        #only keep first 3 channels
        image2 = image[:3].transpose((1, 2, 0))
        image2 *= 255
        image_path = path+"/train_image_{}.png".format(index)
        images_paths.append(image_path)
        cv2.imwrite(image_path, image2)
    return images_paths


def plot_pred(pred, val_mask, y, store_path, epoch, image_name, val_img_index):
    if val_mask != '':
        val_mask = val_mask[val_img_index].cpu().detach().numpy().transpose((1, 2, 0))
    prediction = pred[val_img_index].cpu().detach().numpy()
    prediction = prediction.transpose((1, 2, 0))
    prediction = np.clip(prediction, 0, 1)
    prediction *= 255
    prediction = np.array(
        [prediction[:, :, 0], prediction[:, :, 0], prediction[:, :, 0]]
    )
    prediction = prediction.transpose((1, 2, 0))

    res = y[val_img_index].cpu().detach().numpy()
    res = res.transpose(1, 2, 0)
    res = np.clip(res, 0, 1)
    res *= 255
    if val_mask != '':
        res = np.array([res[:, :, 0], res[:, :, 0], val_mask[:, :, 0] * 255])
    else:
        res = np.array([res[:, :, 0],res[:, :, 0],res[:, :, 0]])
    res = res.transpose((1, 2, 0))

    con = np.concatenate((prediction, res), axis=1)
    image_name = image_name[val_img_index]
    image_name = image_name.replace("\\", "/")
    image_name = image_name.split("/")[-1].split(".")[0]
    image_path = "{}/epoch_{:0>5}_img_{}_pred_res.png".format(store_path, epoch, image_name)
    #print("store: {}".format(image_path))
    cv2.imwrite(image_path, con)
    return image_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--img_size", type=int, nargs="+", default=[256])
    parser.add_argument("--dataset", action="store", default="training_dataset")
    parser.add_argument("--validation", action="store", default="")
    parser.add_argument("--input_channels", type=int, nargs="+", action="store", default=[3, 3])
    parser.add_argument("--output_channels", type=int, action="store", default=1)
    parser.add_argument("--experiment_name", action="store", default="default")
    parser.add_argument("--weights", action="store", default="")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.0008)
    parser.add_argument("--device", action="store", type=str, default="0")
    parser.add_argument("--algorithm", action="store", type=str, default="multidimPixelClassifier")  # AttentionPixelclassifier
    parser.add_argument("--no_loss_mask", action="store_true", default=False)
    parser.add_argument("--test_dataset", action="store", default="")
    parser.add_argument('--pfi', action='store_true', help='permutation feature importance flag for test method')
    parser.add_argument("--target_img_name", action="store", default="", help='Provide the file path to the bboxes_X_X.tiff file from the indata folder')
    parser.add_argument("--workdir", action="store", default="./")
    parser.add_argument("--n_channels_per_layer", action="store", default="10,8,6,4,2", help='ex: 10,8,6,4,2')
    parser.add_argument("--n_coefficients_per_upsampling_layer", action="store", default=None, help='ex: 4,4    Note: Must be the correct number of values corresponding to the upsampling layers, i.e. N_values = (len(n_channels_per_layer)/2) - 1')

    opt = parser.parse_args()

    if opt.mlflow:
        # close any existing runs
        mlflow.end_run()
        #mlflow.set_tracking_uri("file:/mnt/mlflow_tracking/mlruns")
        mlflow.set_tracking_uri(os.path.join('file:'+opt.workdir,"mlruns"))
        # client = MlflowClient()
        # experiment = client.get_experiment_by_name(opt.experiment_name)
        # if experiment is not None:
        mlflow.set_experiment(opt.experiment_name)
        # else:
        #     mlflow.create_experiment(name=opt.experiment_name,
        #                              artifact_location=os.path.join(opt.workdir, 'mlflow_artifacts'))

        mlflow.start_run()

        arguments = {}
        for arg in opt.__dict__:
            if opt.__dict__[arg] is not None:
                arguments[arg] = opt.__dict__[arg]
        mlflow.log_params(arguments)

    main(opt)

    if opt.mlflow:
        mlflow.end_run()


# below code is for trouble-shooting purposes only
# from types import SimpleNamespace
#
# # Replace 'example_region' and 'example_configuration' with actual values
# region = 'alpine'
# configuration = '22,33,44,55,44,33,22'
#
# opt = SimpleNamespace(
#     epochs=100,
#     batch_size=14,
#     img_size=[128],
#     dataset=f"data/processed_geodata/{region}/{region}_geodata",
#     validation=f"data/processed_geodata/{region}/{region}_geodata/validation",
#     input_channels=[11],  # Assuming '11' is the correct value
#     output_channels=1,
#     experiment_name=f"{region}_flex_{configuration}",
#     weights="",
#     plot=True,
#     mlflow=False,
#     learning_rate=0.0008,
#     device="gpu",
#     algorithm="AttentionPixelClassifierFlex",
#     no_loss_mask=False,
#     test_dataset=f"data/processed_geodata/{region}/{region}_geodata/testset/",
#     pfi=True,
#     target_img_name="",
#     workdir="/Users/toban562/Projects/bioscann",
#     n_channels_per_layer=configuration,  # Replace with actual configuration
#     n_coefficients_per_upsampling_layer=None
# )

