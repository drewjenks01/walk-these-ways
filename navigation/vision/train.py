from navigation.vision.vision_model import VisionModel
from navigation import constants
from navigation.vision.utils.model_utils import get_predicted_class
from navigation.vision.utils.data_processing_utils import (
    make_dataloaders_from_demo,
    load_demo_data,
)

import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import datetime
import pickle as pkl
import plotly.express as px
import pandas as pd
import numpy as np


def save_model_and_info(model, info, plots):
    now = datetime.datetime.now()
    # dd/mm/YY_H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    path = constants.TRAINED_MODEL_PATH / dt_string
    path.mkdir()
    logging.info(f"Model path: {path}")

    # save txt file with all info easily readable
    with open(path / "info.txt", "w") as f:
        for k, v in info.items():
            f.writelines(f"{k} = {v}\n")
    logging.info("Text file saved")

    # save pkl file with info dict
    with open(path / "info.pkl", "wb") as f:
        pkl.dump(info, f)
    logging.info("Info dict saved")

    # save model
    torch.save(model, path / "model.pth")
    logging.info("Model saved")

    # save model plots
    for name,plot in plots.items():
        plot.write_image(path / f"{name}.png")
    logging.info("Training plots saved")

def confusion_matrix(actual, predicted):

    # extract the different classes
    classes = range(constants.NUM_GAITS)

    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for true_label, predicted_label in zip(actual, predicted):
        confmat[true_label][predicted_label] += 1

    return confmat

def train(args):
    # Define your training parameters
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    train_perc = args.train_perc
    demo_folder = args.demo_folder
    demo_name = args.demo_name
    num_classes = constants.NUM_GAITS

    # Instantiate the model
    model = VisionModel(inference_type="offline", num_cameras=constants.NUM_CAMERAS)
    print(constants.DEVICE)
    model = model.to(constants.DEVICE)  # move the model to the GPU
    model.train()  # Set the model to training mode

    # make dataloaders
    demo_path = constants.DEMO_BASE_PATH / demo_folder / demo_name
    demo_data = load_demo_data(demo_path=demo_path)
    train_loader, test_loader = make_dataloaders_from_demo(
        demo_data=demo_data, batch_size=batch_size, train_perc=train_perc
    )

    # Define loss functions for classification and regression
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

    # Define optimizer for classification and regression heads only
    optimizer = optim.Adam(
        [
            {"params": model.gait_head.parameters()},
            {"params": model.command_head.parameters()},
        ],
        lr=learning_rate,
    )

    # Training loop
    logging.info("Starting train loop")
    loss_values = {"classification": [], "regression": []}
    for epoch in range(num_epochs):
        running_classification_loss = 0.0
        running_regression_loss = 0.0

        for images, labels, regression_targets in train_loader:
            images = images.float()
            labels = labels.type(torch.LongTensor)
            regression_targets = regression_targets.float()

            images = images.to(constants.DEVICE)
            labels = labels.to(constants.DEVICE)
            regression_targets = regression_targets.to(constants.DEVICE)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            class_logits, regression_outputs = model(images)

            # Compute classification loss
            classification_loss = classification_criterion(
                class_logits.view(-1, num_classes), labels.view(-1)
            )

            # Compute regression loss
            regression_loss = regression_criterion(
                regression_outputs, regression_targets
            )

            # Total loss (you can adjust the weights for each loss if needed)
            total_loss = classification_loss + regression_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            running_classification_loss += classification_loss.item()
            running_regression_loss += regression_loss.item()

        # Print epoch statistics
        average_classification_loss = running_classification_loss / len(train_loader)
        average_regression_loss = running_regression_loss / len(train_loader)
        loss_values["classification"].append(average_classification_loss)
        loss_values["regression"].append(average_regression_loss)
        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] Classification Loss: {average_classification_loss:.4f}, Regression Loss: {average_regression_loss:.4f}"
        )

    logging.info("Training finished")

    logging.info("Testing model")
    model.eval()  # Set the model to evaluation mode

    testing_classification_loss = 0.0
    testing_regression_loss = 0.0

    # truth, pred
    test_info = {
        "classification": [[], []],
        "regression": [[[], []], [[], []], [[], []]],
    }
    with torch.no_grad():
        for images, labels, regression_targets in test_loader:
            images = images.float()
            labels = labels.type(torch.LongTensor)
            regression_targets = regression_targets.float()

            images = images.to(constants.DEVICE)
            labels = labels.to(constants.DEVICE)
            regression_targets = regression_targets.to(constants.DEVICE)

            # Forward pass during testing
            class_logits, regression_outputs = model(images)

            class_preds = get_predicted_class(class_logits)
            test_info["classification"][0] += labels.flatten().cpu()
            test_info["classification"][1] += class_preds.flatten().cpu()

            test_info["regression"][0] += regression_targets.flatten().cpu()
            test_info["regression"][1] += regression_outputs.flatten().cpu()

            # Compute testing classification loss
            testing_classification_loss += classification_criterion(
                class_logits.view(-1, num_classes), labels.view(-1)
            ).item()

            # Compute testing regression loss
            testing_regression_loss += regression_criterion(
                regression_outputs, regression_targets
            ).item()

    # Calculate and print testing statistics
    average_testing_classification_loss = testing_classification_loss / len(test_loader)
    average_testing_regression_loss = testing_regression_loss / len(test_loader)
    logging.info(
        f"Testing Classification Loss: {average_testing_classification_loss:.4f}, Testing Regression Loss: {average_testing_regression_loss:.4f}"
    )

    logging.info("Saving model, train, and test info")
    save_info = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "train_perc": train_perc,
        "demo_folder": demo_folder,
        "demo_name": demo_name,
    }
    save_info["final_train_loss"] = [
        average_classification_loss,
        average_regression_loss,
    ]
    save_info["final_test_loss"] = [
        average_testing_classification_loss,
        average_testing_regression_loss,
    ]

    # make training curve plots
    # Create a DataFrame for the classification training curve data
    classification_train_curve_data = pd.DataFrame(
        {"Epoch": range(num_epochs), "Loss": loss_values["classification"]}
    )

    # Create a training curve plot for classification
    fig_classification_train_curve = px.line(
        classification_train_curve_data,
        x="Epoch",
        y="Loss",
        title="Classification Training Curve",
    )

    # Create a DataFrame for the regression training curve data
    regression_train_curve_data = pd.DataFrame(
        {"Epoch": range(num_epochs), "Loss": loss_values["regression"]}
    )

    # Create a training curve plot for regression
    fig_regression_train_curve = px.line(
        regression_train_curve_data,
        x="Epoch",
        y="Loss",
        title="Regression Training Curve",
    )

    # Create a scatter plot for classification
    cm = confusion_matrix(test_info['classification'][0], test_info['classification'][1])
    fig_test_confusion = px.imshow(cm,
                labels=dict(x="Predicted", y="Actual"),
                x=constants.GAIT_NAMES,  # Replace with your class labels
                y=constants.GAIT_NAMES,  # Replace with your class labels
                title="Confusion Matrix",
                text_auto=True)

    # Customize the layout (optional)
    fig_test_confusion.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis=dict(side="top"),
    )

    plots = {
        "classification_training_curve": fig_classification_train_curve,
        "regression_training_curve": fig_regression_train_curve,
        "test_confusion": fig_test_confusion,
    }
    save_info["test_classification"] = testing_classification_loss
    save_info["test_regression"] = testing_regression_loss

    save_model_and_info(model=model, info=save_info, plots=plots)

def parse_args():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train a full vision model.")

    parser.add_argument("--demo_folder", type=str, required=True)
    parser.add_argument("--demo_name", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train_perc", type=float, default=0.8)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
