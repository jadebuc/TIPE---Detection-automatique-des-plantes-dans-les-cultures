import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="data.yaml", epochs=50)


