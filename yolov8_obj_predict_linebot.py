## ========================================
##
##      物件偵測 - LineBot 版 - 照片預測
##
## ========================================

from ultralytics import YOLO


def predict(img, model, conf=0.8, device="cpu"):
    """
    接收一張照片與一個模型權重
    輸出一個 類別名稱(ex: A01N) / 無法辨識 (資料型態: str)

    img: 照片
    weight: 模型權重路徑(xxx/xxx.pt)   ->  model: 已經讀取參數的模型
    conf: 閾值(預設 0.8)
    device: 使用硬體("cpu"(預設) / "gpu")
    """

    # 讀取 yolov8 模型權重
    # model = YOLO(weight) 
    # 圖片辨識
    results = model.predict(img, conf=conf, device=device)  # (圖片, 信心值, 設備)

    # 取得預測結果
    boxes_data = results[0].boxes.cpu()     # 取得需要的資料，並移到 cpu 運行
    name_dict =results[0].names             # 取得訓練資料各類別與編號的對應
    class_name = boxes_data.cls.numpy()     # 取得偵測到的物件

    # 取得偵測到的類別名稱
    if len(class_name) == 0:
        pre_name = "無法辨識"
    else:
        pre_name = name_dict[int(class_name[0])]
    
    return pre_name


