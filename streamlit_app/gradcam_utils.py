import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
import os

def generate_and_save_gradcam(model, img_array, last_conv_layer_name, save_path='output_image/gradcam.jpg'):
    # Check if layer exists
    if last_conv_layer_name not in [layer.name for layer in model.layers]:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in model. Available layers: {[layer.name for layer in model.layers]}")

    # Create sub-model
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Forward pass + gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize safely
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        max_val = tf.constant(1e-10)
    heatmap /= max_val

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_image = np.uint8(img_array[0] * 255.0)
    if original_image.shape[-1] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    elif original_image.shape[-1] == 3:
        pass
    else:
        raise ValueError("Unexpected image shape: ", original_image.shape)

    # Overlay heatmap
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Save image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, superimposed_img)

    confidence = float(tf.reduce_max(predictions[0]))
    return save_path, int(pred_index), confidence
