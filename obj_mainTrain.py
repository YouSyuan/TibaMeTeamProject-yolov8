import os, cv2, datetime
from ultralytics import YOLO
import multiprocessing
from PIL import Image
import cv2

# C:\Users\\AppData\Roaming\Ultralytics


def main():
    # 加载模型
    # model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
    model = YOLO("yolov8s.pt")
    # print(model.info(detailed=True))
    # 使用模型
    model.train(data='config.yaml', 
                imgsz=(512,288),
                epochs=70,
                patience=10,
                batch=48,
                optimizer="Adam",
                # cache="ram",
                device=0,                
      )


if __name__ == '__main__':
    start = datetime.datetime.now()

    multiprocessing.freeze_support()
    main()

    end = datetime.datetime.now()
    print("花費：", (end-start))
