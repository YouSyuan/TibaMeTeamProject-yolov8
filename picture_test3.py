## ========================================
##
##            照片預測 for 多張
##
## ========================================

from ultralytics import YOLO
import multiprocessing
from PIL import Image
import glob

# from class_name import CLASSES_NAME

CLASSES_NAME = ['A01N', 'A02W']
img_path = []
for i in CLASSES_NAME:
    img_path += glob.glob(f"H:/TibaMe_TeamProject/images_data/test_img/{i}/*")

img_open = [Image.open(x) for x in img_path]
model = YOLO("runs/detect/train5/weights/best.pt")

results = model.predict(img_open[0], conf=0.5, device=0)
boxes_d = results[0].boxes.cpu()
print(boxes_d)