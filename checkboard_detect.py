import cv2
import tensorflow


def get_border(src):
    blur = 1
    (maxb, maxg, maxr, minb, ming, minr) = (172, 185, 218, 118, 150, 178)
    h, w = src.shape[:2]
    tmp = cv2.resize(src, (int(w * 0.2), int(h * 0.2)));
    height, width = tmp.shape[:2]
    while blur <= 15:
        tmp = cv2.medianBlur(tmp, blur)
        (maxi, maxj, mini, minj) = (0, 0, int(height), int(width))
        for i in range(height):
            for j in range(width):
                (b, g, r) = tmp[i, j]
                if minb < b < maxb and ming < g < maxg and minr < r < maxr:
                    if i < mini:
                        mini = i
                    if i > maxi:
                        maxi = i
                    if j < minj:
                        minj = j
                    if j > maxj:
                        maxj = j
        maxi *= 5
        mini *= 5
        maxj *= 5
        minj *= 5
        if (435 < (maxi - mini) < 735) and (435 < (maxj - minj) < 735):
            return maxi, mini, maxj, minj
        blur += 2
    return 0, 0, 0, 0


def checkboard_detect(src):
    (maxi, mini, maxj, minj) = get_border(src)
    if maxi == mini:
        print("无法识别棋盘，或者棋盘不存在")
        return -1
    ret = src[mini:maxi, minj:maxj]
    return ret
