# 快速产生数据集以备使用
import random
import pygame

from data_structure import Position
from gobang_system import Gobang

def shuffled_coords(min, max):
    coords = []
    color = 0
    for pos_x in range(15):
        for pos_y in range(15):
            coords.append({'type': color, 'coord': Position(pos_x, pos_y)})
            color = random.randint(0, 1)
    random.shuffle(coords)
    number = random.randint(min, max)
    return coords[:number]


def generate_checkboard(num, min = 10, max = 30):
    random.seed()
    gobang = Gobang()
    pygame.init()
    size = width, height = 535, 535
    screen = pygame.display.set_mode(size, 0, 32)
    for i in range(num):
        gobang.init()
        gobang.drop_by_coords(shuffled_coords(min, max))
        gobang.save_image(i)
        gobang.save_coord(i)