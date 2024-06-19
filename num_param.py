from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random

config = py2cfg('config/loveda/unetformer.py')
net = config.net

print("GeoSeg : ", sum(p.numel() for p in net.parameters() if p.requires_grad))
print(net)
print("Backbone : ", sum(p.numel() for p in net.backbone.parameters() if p.requires_grad))
print("decoder : ", sum(p.numel() for p in net.decoder.parameters() if p.requires_grad))