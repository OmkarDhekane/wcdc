import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras

"""
Grad-CAM utilities for the model:

input_layer_1 -> densenet121 -> global_average_pooling2d
               -> dropout -> dense -> dropout_1 -> dense_1 (7 classes)
"""

def make_gradcam_heatmap(
    img_array,
    model: keras.Model,
    backbone_name: str = "densenet121",
    class_index: int | None = None,
):
    """
    Compute Grad-CAM heatmap for a given class.

    Parameters
    ----------
    img_array : np.ndarray
        Shape (1, H, W, 3), preprocessed (float32, [0, 1]).
    model : keras.Model
        Full classification model (backbone + head).
    backbone_name : str
        Name of the backbone layer in the model (here: "densenet121").
    class_index : int or None
        Target class index. If None, the top predicted class is used.

    Returns
    -------
    heatmap : np.ndarray
        2D array (Hc, Wc) with values in [0, 1].
    """

    # Ensure tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Grab layers we need
    backbone = model.get_layer(backbone_name)           # densenet121
    gap = model.get_layer("global_average_pooling2d")
    drop1 = model.get_layer("dropout")
    dense = model.get_layer("dense")
    drop2 = model.get_layer("dropout_1")
    out_dense = model.get_layer("dense_1")              # final 7-logit layer

    with tf.GradientTape() as tape:
        conv_outputs = backbone(img_tensor)             # (1, 7, 7, 1024)
        tape.watch(conv_outputs)                        # track for gradients

        # 2) Manually pass through head
        x = gap(conv_outputs)
        x = drop1(x, training=False)                    # disable dropout randomness
        x = dense(x)
        x = drop2(x, training=False)
        preds = out_dense(x)                            # (1, num_classes)

        # 3) Pick class index
        if class_index is None:
            class_index = tf.argmax(preds[0])

        class_score = preds[:, class_index]             # (1,)

    # 4) Gradient of class score w.r.t conv feature maps
    grads = tape.gradient(class_score, conv_outputs)    # (1, 7, 7, 1024)

    # 5) Global average pool the gradients over spatial dims -> (1024,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6) Weight conv feature maps by pooled grads
    conv_outputs = conv_outputs[0]                      # (7, 7, 1024)
    heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=([2], [0]))  # (7, 7)

    # 7) ReLU + normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(heatmap, image, alpha: float = 0.4):
    """
    Overlay a Grad-CAM heatmap on top of a PIL image.

    Parameters
    ----------
    heatmap : np.ndarray
        2D array in [0, 1].
    image : PIL.Image.Image
        Original RGB image.
    alpha : float
        Blend factor between image and heatmap.

    Returns
    -------
    superimposed_img : np.ndarray
        RGB uint8 image with overlay.
    """
    # Resize heatmap to match original image size (width, height)
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)

    # Apply color map and convert BGR -> RGB
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    # Blend with original image
    img = np.array(image)
    superimposed_img = img * (1 - alpha) + jet * alpha
    superimposed_img = np.uint8(superimposed_img)

    return superimposed_img
