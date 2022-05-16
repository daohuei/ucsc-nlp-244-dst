import os
from datetime import datetime
import copy
import torch
import logging

logging.basicConfig()


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def get_config(args):
    config = copy.copy(args)
    config.writer = None
    return config
