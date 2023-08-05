import datetime, multiprocessing, os
from ultralytics import YOLO



def main(data):
    # Load a model
    model = YOLO("yolov8n-cls.pt")

    # Train
    results = model.train(data=data, epochs=170, batch=32, imgsz=512, device=0)


if __name__ == '__main__':  
    start = datetime.datetime.now()  

    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    multiprocessing.freeze_support()      
    DATA = "H:/TibaMe_TeamProject/projects/yolov8_classification/data"
    main(DATA)


    end = datetime.datetime.now()
    print("花費：", (end-start))