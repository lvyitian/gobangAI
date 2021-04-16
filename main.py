import pickle

import cv2

from game import game
from train import *
from make_trains import make_trains
from checkboard_detect import checkboard_detect
from game_with_ai import game_ai
from data_structure import *

def main():
    # src = cv2.imread("image\\detect\\image3.jpg")
    # src = checkboard_detect(src)
    # cv2.imwrite("out\\image_detect3.jpg", src)
    game_ai()
    # for i in range(10000):
    #     data_sl.save(shuffled_coords(20, 60), "test\\coords\\coord" + str(i) + ".dat")
    #     if i % 100 == 0:
    #         print(i)

main()