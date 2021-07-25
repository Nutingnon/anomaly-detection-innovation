import os
import pandas as pd
import numpy as np
from dataloader import loader_parkinson
import re

X,y = loader_parkinson.read_data(loader_parkinson.file_path)