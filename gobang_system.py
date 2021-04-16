import sys

import pygame
import pickle
from data_structure import *
from  game_tree import create_cbmap

class Gobang(object):
    background_filename = 'image\\chessboard.png'
    white_chessball_filename = 'image\\whiteChess.png'
    black_chessball_filename = 'image\\blackchess.png'
    top, left, space, lines = (23, 23, 35, 15)  # 棋盘格子位置相关

    def init(self):
        try:
            self._chessboard = pygame.image.load(self.background_filename)
            self._white_chessball = pygame.transform.scale(pygame.image.load(self.white_chessball_filename).convert_alpha(), (int(self.space * 0.8),int(self.space * 0.8)))
            self._black_chessball = pygame.transform.scale(pygame.image.load(self.black_chessball_filename).convert_alpha(), (int(self.space * 0.8),int(self.space * 0.8)))
            self.font = pygame.font.SysFont('arial', 16)
            self.ball_rect = self._white_chessball.get_rect()
            self.points = [[] for i in range(self.lines)]
            self.black_turn = True  # 黑子先手
            self.ball_coord = []  # 记录黑子和白子逻辑位置
            self.cbmap = create_cbmap()
            for i in range(self.lines):
                for j in range(self.lines):
                    self.points[i].append(Position(self.left + i * self.space, self.top + j * self.space))
        except pygame.error as e:
            sys.exit()

    def chessboard(self):
        return self._chessboard

    # 在(i,j)位置落子    
    def drop_at(self, i, j):
        pos_x = self.points[i][j].x - int(self.ball_rect.width / 2)
        pos_y = self.points[i][j].y - int(self.ball_rect.height / 2)

        ball_pos = {'type': 0 if self.black_turn else 1, 'coord': Position(i, j)}
        if self.black_turn:  # 轮到黑子下
            self._chessboard.blit(self._black_chessball, (pos_x, pos_y))
        else:
            self._chessboard.blit(self._white_chessball, (pos_x, pos_y))

        self.ball_coord.append(ball_pos)  # 记录已落子信息
        self.cbmap[i][j] = -1 if self.black_turn else 1
        self.black_turn = not self.black_turn  # 切换黑白子顺序

    # 指定落子颜色在(i,j)位置落子
    def drop_color_at(self, i, j, color):
        pos_x = self.points[i][j].x - int(self.ball_rect.width / 2)
        pos_y = self.points[i][j].y - int(self.ball_rect.height / 2)

        ball_pos = {'type': 1 if color else 0, 'coord': Position(i, j)}
        if not color:  # 轮到黑子下
            self._chessboard.blit(self._black_chessball, (pos_x, pos_y))
        else:
            self._chessboard.blit(self._white_chessball, (pos_x, pos_y))

        self.ball_coord.append(ball_pos)  # 记录已落子信息

    def drop_by_coords(self, coords):
        for i in coords:
            self.drop_color_at(i['coord'].x, i['coord'].y, i['type'])

    # 判断是否已产生胜方
    def check_over(self):
        if len(self.ball_coord) > 8:  # 只有黑白子已下4枚以上才判断
            direct = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、斜、反斜 四个方向检查
            for d in direct:
                if self._check_direct(d):
                    return True
        return False

    # 判断最后一个棋子某个方向是否连成5子，direct:(1,0),(0,1),(1,1),(1,-1)
    def _check_direct(self, direct):
        dt_x, dt_y = direct
        last = self.ball_coord[-1]
        line_ball = []  # 存放在一条线上的棋子
        for ball in self.ball_coord:
            if ball['type'] == last['type']:
                x = ball['coord'].x - last['coord'].x
                y = ball['coord'].y - last['coord'].y
                if dt_x == 0:
                    if x == 0:
                        line_ball.append(ball['coord'])
                        continue
                if dt_y == 0:
                    if y == 0:
                        line_ball.append(ball['coord'])
                        continue
                if x * dt_y == y * dt_x:
                    line_ball.append(ball['coord'])

        if len(line_ball) >= 5:  # 只有5子及以上才继续判断
            sorted_line = sorted(line_ball)
            for i, item in enumerate(sorted_line):
                index = i + 4
                if index < len(sorted_line):
                    if dt_x == 0:
                        y1 = item.y
                        y2 = sorted_line[index].y
                        if abs(y1 - y2) == 4:  # 此点和第5个点比较y值，如相差为4则连成5子
                            return True
                    else:
                        x1 = item.x
                        x2 = sorted_line[index].x
                        if abs(x1 - x2) == 4:  # 此点和第5个点比较x值，如相差为4则连成5子
                            return True
                else:
                    break
        return False

    # 检查(i,j)位置是否已占用    
    def check_at(self, i, j):
        for item in self.ball_coord:
            if (i, j) == item['coord']:
                return False
        return True

    # 通过物理坐标获取逻辑坐标        
    def get_coord(self, pos):
        x, y = pos
        i, j = (0, 0)
        oppo_x = x - self.left
        if oppo_x > 0:
            i = round(oppo_x / self.space)  # 四舍五入取整
        oppo_y = y - self.top
        if oppo_y > 0:
            j = round(oppo_y / self.space)
        return i, j

    def save_image(self, count = 0):
        filename = "out\\image" + str(count) + ".jpg"
        pygame.image.save(self._chessboard, filename)

    def save_coord(self, count = 0):
        filename = "out\\coord" + str(count) + ".dat"
        data_sl.save(self.ball_coord, filename)

    def load_coord(self, count = 0):
        filename = "out\\coord" + str(count) + ".dat"
        data = data_sl.load(filename)
        self.drop_by_coords(data)

    def check_full(self):
        for i in self.cbmap:
            for j in i:
                if j == 0:
                    return False
        return True