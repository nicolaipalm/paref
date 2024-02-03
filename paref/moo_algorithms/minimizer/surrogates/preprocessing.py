import numpy as np


def preprocess_x(x: np.ndarray, data: np.ndarray):
    min_components = np.min(data, axis=0)
    max_components = np.max(data, axis=0)
    return (x - min_components) / (max_components - min_components)  # normalize to [0,1]


def postprocess_x(x_processed: np.ndarray, data: np.ndarray):
    min_components = np.min(data, axis=0)
    max_components = np.max(data, axis=0)
    return x_processed * (max_components - min_components) + min_components  # denormalize from [0,1]


def preprocess_y(y: np.ndarray, data: np.ndarray):
    min_components = np.min(data, axis=0)
    max_components = np.max(data, axis=0)
    return (y - min_components) / (max_components - min_components)  # normalize to [0,1]


def postprocess_y(y_processed: np.ndarray, data: np.ndarray):
    min_components = np.min(data, axis=0)
    max_components = np.max(data, axis=0)
    return y_processed * (max_components - min_components) + min_components  # denormalize from [0,1]


def postprocess_std(std: np.ndarray, data: np.ndarray):
    min_components = np.min(data, axis=0)
    max_components = np.max(data, axis=0)
    return std * (max_components - min_components)  # denormalize from [0,1]
