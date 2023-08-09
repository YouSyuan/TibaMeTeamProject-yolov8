## ========================================
##
##            照片預測 for 單張
##
## ========================================

from ultralytics import YOLO
import multiprocessing
from PIL import Image
import cv2
import glob

from class_name import CLASSES_NAME

# classes_name = ['A01N', 'A02W', 'A02E']

def pre(images, model, c, device="cpu"):
    # model = YOLO(model_path)

    # img = Image.open(images)

    results = model.predict(images, conf=c, device=device, batch=1)
    # boxes_data = []
    # boxes_data.append(results[0].boxes)
    # class_name = [boxes_data[0].names]
    boxes_d = results[0].boxes.cpu()
    name_dict = results[0].names
    class_name = boxes_d.cls.numpy()

    if len(class_name) == 0:
        pre_name = ""
    else:
        class_idx = int(class_name[0])
        pre_name = name_dict[class_idx]

    file_name = images.split("\\")[-1]

    if pre_name == file_name[0:4]:
        correct = 1
        # print(f"預測類別：{pre_name}  正確類別：{ans_name}  預測樣本：{images}")
        return (correct, "True", pre_name)
    else:
        if len(pre_name) == 0:
            pre_name = "None"
        correct = 0
        # print(f"預測類別：{pre_name}  正確類別：{ans_name}  預測樣本：{images}")
        return (correct, file_name, pre_name)
    







if __name__ == "__main__":
    weight = "runs/detect/train_no_SS_2_Adam/weights/best.pt"
    model_path = "runs/detect/train_no_SS_2_Adam"
    images_list = glob.glob("test_SS/images/*")    
    txt_name = "預測1"
    conf = 0.8
    model = YOLO(weight)

    total_samples = 0
    total_correct = 0
    err_classes = {}

    for n, img in enumerate(images_list):
        print(n,"=====", img, "================================================")
        correct, file_name, pre_name = pre(img, model, conf, 0)
        total_samples += 1
        total_correct += correct
        if file_name != "True":
            ans = file_name[0:4]
            err_classes.setdefault(ans, [0])
            err_classes[ans][0] += 1
            err_classes[ans].append([file_name, pre_name])
        print()
        
    accuracy = total_correct / total_samples
    print("測試集樣本數:", total_samples)
    print("預測正確數:", total_correct)
    print("預測錯誤數:", total_samples - total_correct)
    print(f"測試集的準確率：{round(accuracy*100, 2)}%")
    
        # 將結果儲存到 result.txt 檔案中
    with open(f'{model_path}/{txt_name}.txt', 'w') as file:
        file.write(f"測試集樣本數: {total_samples}\n")
        file.write(f"預測正確數: {total_correct}\n")
        file.write(f"預測錯誤數: {total_samples - total_correct}\n")
        file.write(f"測試集的準確率：{round(accuracy*100, 2)}%\n")
        file.write("\n")
        file.write("== 錯誤類別 ==\n")
        for key, value in err_classes.items():
            file.write(f"正確類別：{key}, 錯誤數量：{value[0]}\n")
            n = 0
            for i in value[1:]:
                file.write(f"{i[0]} -> {i[1]},\t")
                n += 1
                if n == 4:
                    n = 0
                    file.write("\n")
            file.write("\n")


