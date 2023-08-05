## ========================================
##
##               照片處理工具
##
## ========================================

from PIL import Image


def filling(img, size, open=False, save=False, save_path="", back_ground=(0,0,0)):
    """ 
    將圖片填充成正方形 
    調整成 size * size
    """

    if open:
        img = Image.open(img)

    # 取得原始圖片的尺寸
    width, height = img.size

    # 找出較大的一邊
    max_size = max(width, height)

    # 新建一個空白的正方形圖片
    square_img = Image.new('RGB', (max_size, max_size), back_ground)

    # 計算將原始圖片放入正方形中的位置
    paste_x = (max_size - width) // 2
    paste_y = (max_size - height) // 2

    # 將原始圖片貼上正方形中
    square_img.paste(img, (paste_x, paste_y))

    # 調整大小成為目標大小
    img = square_img.resize((size, size))

    # 儲存圖片
    if save:
        img.save(save_path)
    
    return img


def cutting(img, prop=0.4, place=1, open=False, save=False, save_path=""):
    """ 
    裁切圖片 
    0.裁上方  1.裁中間  2.裁下面  其他:原圖
    """

    if open:
        img = Image.open(img)
        
    w, h = img.size
    h_resize = int(h * prop)
    if place == 0:
        img =  img.crop((0, 0, w, h_resize))  # (左上角座標x1,y1,右下角座標x2,y2)
    elif place == 1:
        y = int(h * ((1-prop) / 2))
        img =  img.crop((0, y, w, y+h_resize))
    elif place == 2:
        y = int(h * (1-prop))
        img =  img.crop((0, y, w, h))

    
    if save:
        img.save(save_path)
    
    return img
    






if __name__ == "__main__":
    import glob, os

    images_list = glob.glob(f"H:/TibaMe_TeamProject/projects/img_no_SS/test/*/*")
    save_path = "H:/TibaMe_TeamProject/projects/img_no_SS"
    for i in images_list:
        name = i.split("\\")
        print(name[2])
        os.makedirs(f"{save_path}/test2/{name[1]}", exist_ok=True)
        img = Image.open(i)
        img = cutting(img, prop=0.4, place=1)
        filling(img, 512, save=True, save_path=f"{save_path}/test2/{name[1]}/{name[2]}", back_ground=(0,0,0))