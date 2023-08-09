## ========================================
##
##      影像分類 - LineBot 版 - 照片預測
##
## ========================================

from PIL import Image
# yolov8
from ultralytics import YOLO
# my tools function
from img_tools import *



def predict(img, model, conf=0.8, open=False, cut=False, fil=False, device="cpu"):
    """
    接收一張照片與一個模型權重
    輸出一個 類別名稱(ex: A01N) / 無法辨識(信心度<conf)  (資料型態: str)

    img: 照片
    weight: 模型權重路徑(xxx/xxx.pt)  ->  model: 已經讀取參數的模型
    conf: 閾值(預設 0.8)
    cut: 裁切到片
    fil: 將裁切好照片填補成正方形
    device: 使用硬體("cpu"(預設) / 0 (顯卡, type:int))
    """
    # model = YOLO(weight)
    # if open:
    #     img = Image.open(img)
    if cut:   # 取得路牌的部分
        img = cutting(img, 0.4, 1)
    if fil:   # 填充成正方形  
        img = filling(img, 512)
    # 預測
    results = model.predict(img)
    names_dict = results[0].names  # 取得模型中包含各類別的編號(字典{0:A01N,...})
    top1 = results[0].probs.top1  # 取得機率最高(類別)的編號
    top1_probs = round(results[0].probs.top1conf.item(), 4)  # 取得該類別的機率(float)
    top1_name = names_dict[top1]  # 取得該類別名稱

    if top1_probs < conf:
        return "無法辨識" 
        
    return top1_name
    
