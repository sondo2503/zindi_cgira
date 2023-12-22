from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
import tqdm

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import get_dataloader
from transform import get_transform
from models import get_model
from losses import get_loss
from optimizer import get_optimizer
from scheduler import get_scheduler
import utils.config
import utils.checkpoint


def run(config, checkpoint_name, images):
    model = get_model(config).cuda()

    checkpoint = utils.checkpoint.get_checkpoint(config, name=checkpoint_name)
    utils.checkpoint.load_checkpoint(model, None, checkpoint)
    images = torch.from_numpy(images)
    images.cuda()
    B, T, C, H, W = images.shape
    logits = model(images.view(-1, C, H, W))[:, :5]
    logits = logits.view(B, T, -1)
    probabilities = F.sigmoid(logits)
    probabilities = probabilities.mean(dim=1)
    return probabilities


image = cv2.imread('/root/sondv/cgiar/data/images/7f9714250c3d2aedb9bff60bbd0c39c40923da12.jpg')
x = run(config=utils.config._get_default_config(), checkpoint_name='epoch_0000.pth', images=image)
print(x)