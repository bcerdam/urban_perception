import utils
import torch
import os
import numpy as np
from raw_feat import RawFeatInference
from raw_feat_reg import RawFeatRegInference
from torchvision import models


def validate_model(weight_path, pp2_validation, m_w, m_t, similarity_threshold, inference_model):
    utils.metrics(pp2_validation, inference_model, m_w, m_t, similarity_threshold, weight_path, None)