## ========================================
##
##               計算資料數量
##
## ========================================


import glob


path = "data/train/*/*"
path1 = "data/val/*/*"
path2 = "data/*/*/*"



print(len(glob.glob(path)))
print(len(glob.glob(path1)))
print(len(glob.glob(path2)))