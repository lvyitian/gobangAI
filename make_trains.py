import cv2

from data_structure import *
from checkboard_detect import checkboard_detect


# 把棋盘图片分为225个小块
def split_checkboard(src):
    tmp = cv2.resize(src, (535, 535))
    side, space = 5, 35
    pieces = []
    for i in range(15):
        for j in range(15):
            minx = side + space * i
            miny = side + space * j
            pieces.append(tmp[minx:minx + space, miny:miny + space])
    return pieces


def calc_pieces(pieces):
    values = []
    for k in pieces:
        sumb, sumg, sumr = 0, 0, 0
        for i in k:
            for j in i:
                b, g, r = j
                sumb += b
                sumg += g
                sumr += r
        div = 255 * 35 * 35
        sumb /= div
        sumg /= div
        sumr /= div
        values.append({'b': sumb, 'g': sumg, 'r': sumr})
    return values


def values_to_training_set(values, coords):
    trains = values[:]
    for coord in coords:
        x, y = coord['coord']
        trains[y * 15 + x]['type'] = 1 if coord['type'] else -1
    for i in trains:
        if 'type' not in i.keys():
            i['type'] = 0
    return trains


# 从截图和坐标集到训练集
def make_trains(src, coords):
    ret = checkboard_detect(src)
    pieces = split_checkboard(ret)
    values = calc_pieces(pieces)
    trains = values_to_training_set(values, coords)
    return trains


def make_tests(src):
    ret = checkboard_detect(src)
    pieces = split_checkboard(ret)
    values = calc_pieces(pieces)
    return values
