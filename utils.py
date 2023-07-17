import numpy as np
from torch import nn
import os


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md
def weight_histograms_conv2d(writer, step, weights, layer_number):
  weights_shape = weights.shape
  num_kernels = weights_shape[0]
  for k in range(num_kernels):
    flattened_weights = weights[k].flatten()
    tag = f"layer_{layer_number}/kernel_{k}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, layer_number):
  flattened_weights = weights.flatten()
  tag = f"layer_{layer_number}"
  writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms(writer, step, model):
#   print("Visualizing model weights...")
  # Iterate over all model layers
#   import pdb;pdb.set_trace()
  for layer_num, (name, value) in enumerate(model.named_parameters()):
    # Get layer
    # layer = model.layers[layer_number]
    # Compute weight histograms for appropriate layer
    # if isinstance(layer, nn.Conv2d):
    #   weights = layer.weight
    #   weight_histograms_conv2d(writer, step, weights, layer_number)
    # elif isinstance(layer, nn.Linear):
    weights = value
    weight_histograms_linear(writer, step, value.cpu().detach().numpy(), name)