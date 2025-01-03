# Copyright 2021 Vittorio Mazzia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import logistic

from utils.transformer import Patches


def plotImages(images_batch, img_n, classes):
    """
    Take as input a batch from the generator and plt a number of images equal to img_n
    Default columns equal to max_c. At least inputs of batch equal two
    """
    max_c = 5

    if img_n <= max_c:
        r = 1
        c = img_n
    else:
        r = math.ceil(img_n / max_c)
        c = max_c

    fig, axes = plt.subplots(r, c, figsize=(15, 15))
    axes = axes.flatten()
    for img_batch, label_batch, ax in zip(images_batch[0], images_batch[1], axes):
        ax.imshow(img_batch)
        ax.grid()
        ax.set_title("Class: {}".format(classes[label_batch]))
    plt.tight_layout()
    plt.show()


def plotHistory(history):
    """
    Plot the loss and accuracy curves for training and validation
    """
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.5)
    plt.show()


def plotPatches(x, n_images, patch_size):
    """Plot the original image and a patched like version"""
    for i in range(n_images):
        plt.figure(figsize=(4, 4))
        image = x[np.random.choice(range(x.shape[0]))]
        plt.imshow(image.astype("uint8"))
        plt.axis("off")

        patches = Patches(patch_size)(image[None].astype("float32"))

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[0]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            plt.imshow(patch_img.numpy().astype("uint8"))
            plt.axis("off")


def explainGradCam(
    explainer, ax, img, y, model_1, y_pred_1, model_2, y_pred_2, class_names
):
    """
    [Attention Episode]
    Plot GRADCAM of two trained models. It needs an axes with two columns
    """
    data = ([img], None)

    y_predm_1 = np.argmax(y_pred_1)
    y_predm_2 = np.argmax(y_pred_2)

    grid_1 = explainer.explain(data, model_1, class_index=y_predm_1, image_weight=0.8)
    grid_2 = explainer.explain(data, model_2, class_index=y_predm_2, image_weight=0.8)

    ax[0].set_xlabel(
        "Pred: {} {:2.0f}% ({})".format(
            class_names[y_predm_1], 100 * np.max(y_pred_1), class_names[y]
        ),
        color=("blue" if y == y_predm_1 else "red"),
    )

    ax[1].set_xlabel(
        "Pred: {} {:2.0f}% ({})".format(
            class_names[y_predm_2], 100 * np.max(y_pred_2), class_names[y]
        ),
        color=("blue" if y == y_predm_2 else "red"),
    )
    ax[0].imshow(grid_1)
    ax[1].imshow(grid_2)


def plot_misclassified_images(X_test, y_pred, y_test, labels):
    y_pred_arg = (logistic.cdf(y_pred) > 0.5)[..., 0].astype(np.float32)
    errors_indices = np.where(y_pred_arg != y_test)[0]

    max_c = 5

    r = np.ceil(errors_indices.size / max_c).astype(np.int32)
    c = max_c

    fig, axes = plt.subplots(r, c, figsize=(55, 55))
    axes = axes.flatten()
    for e_index, ax in zip(errors_indices, axes):
        ax.imshow(X_test[e_index].astype(np.uint8))
        ax.grid()
        class_pred = logistic.cdf(y_pred[int(e_index)])[0]
        if class_pred < 0.5:
            class_pred = 1 - class_pred
        ax.set_title(
            "Class {}: {:.2%}".format(
                labels[int(y_pred_arg[int(e_index)])], class_pred
            ),
            color="red",
        )
    plt.tight_layout()
    plt.show()
