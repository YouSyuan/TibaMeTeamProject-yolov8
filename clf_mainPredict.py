## ========================================
##
##                 照片預測
##
## ========================================

# yolov8
from ultralytics import YOLO
# my tools function
from img_tools import *



def predict(img, model_path, cut=False, fil=False):
    """
    images:傳入圖片
    """
    model = YOLO(model_path)
    if cut:   # 取得路牌的部分
        print("cutting")
        img = cutting(img, 0.4, 1)
    if fil:   # 填充成正方形
        print("filling")   
        img = filling(img, 512)
    # 預測
    results = model.predict(img)
    names_dict = results[0].names  # 取得模型中包含各類別的編號(字典{0:A01N,...})
    top1 = results[0].probs.top1  # 取得機率最高(類別)的編號
    top1_probs = round(results[0].probs.top1conf.item(), 4)  # 取得該類別的機率(float)
    top1_name = names_dict[top1]  # 取得該類別名稱

    return top1_name, top1_probs  # 返回 類別名稱, 信心度(可能機率?)
    
    

    

    
if __name__ == "__main__":
    import glob
    from PIL import Image
    from class_name import CLASSES_NAME
    model_path = "H:/TibaMe_TeamProject/projects/yolov8_classification/runs/classify/train5"
    weight = f"{model_path}/weights/best.pt"
    txt_name = "預測2_沒裁切填補"
    

    images_list = glob.glob(f"H:/TibaMe_TeamProject/projects/yolov8_classification/test/*/*")  # 存放測試圖片路徑

    total_samples = len(images_list)  # 所有測試樣本數量
    total_correct = 0  # 紀錄預測正確數量
    total_error = 0  # 紀錄預測錯誤數量
    err_classes = {}  # 紀錄預測錯誤的檔案

    n = 1
    for img_path in images_list:
        
        n += 1
        img_name = img_path.split("\\")[2]  # 正在預測的圖片檔案名稱
        class_name = img_path.split("\\")[2].split("_")[0]  # 該圖片的類別
        print(n, "=====",  img_name, "==============================")
        img = Image.open(img_path)
    
        pre_name, pre_prob = predict(img, weight, cut=False, fil=False)

        if pre_name == class_name:
            total_correct += 1
        else:
            total_error += 1
            err_classes.setdefault(class_name, [0])
            err_classes[class_name][0] += 1
            err_classes[class_name].append([img_name, pre_name, pre_prob])
    
    accuracy = total_correct / total_samples
    
    print("測試集樣本數:", total_samples)
    print("預測正確數:", total_correct)
    print("預測錯誤數:", total_error)
    print(f"測試集的準確率：{round(accuracy*100, 2)}%")
    print()
    print("== 錯誤類別 ==")
    for key, value in err_classes.items():
        print(f"正確類別：{key}, 預測錯誤數量：{value[0]}")
        for i in value[1:]:
            print(i[0], "->", i[1], end="\t")
        print()

    # 將結果儲存到 result.txt 檔案中
    with open(f'{model_path}/{txt_name}.txt', 'w') as file:
        file.write(f"測試集樣本數: {total_samples}\n")
        file.write(f"預測正確數: {total_correct}\n")
        file.write(f"預測錯誤數: {total_error}\n")
        file.write(f"測試集的準確率：{round(accuracy*100, 2)}%\n")
        file.write("\n")
        file.write("== 錯誤類別 ==\n")
        for key, value in err_classes.items():
            file.write(f"正確類別：{key}, 預測錯誤數量：{value[0]}\n")
            n = 0
            for i in value[1:]:
                file.write(f"{i[0]} -> {i[1]} {i[2]},\t")
                n += 1
                if n == 4:
                    n = 0
                    file.write("\n")
            file.write("\n")


