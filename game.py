import sys
import pygame

from gobang_system import Gobang



def game(coords=None):
    if coords is None:
        coords = []
    pygame.init()  # pygame初始化
    size = width, height = 535, 535
    screen = pygame.display.set_mode(size, 0, 32)
    pygame.display.set_caption('五子棋')
    font = pygame.font.SysFont("arial", 32)
    clock = pygame.time.Clock()  # 设置时钟
    game_over = False
    gobang = Gobang()  # 核心类，实现落子及输赢判断等
    gobang.init()  # 初始化
    gobang.drop_by_coords(coords)

    while True:
        clock.tick(20)  # 设置帧率
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and (not game_over):
                if event.button == 1:  # 按下的是鼠标左键
                    i, j = gobang.get_coord(event.pos)  # 将物理坐标转换成矩阵的逻辑坐标
                    if gobang.check_at(i, j):  # 检查(i,j)位置能否被占用，如未被占用返回True
                        gobang.drop_at(i, j)  # 在(i,j)位置落子，该函数将黑子或者白子画在棋盘上
                        if gobang.check_over():  # 检查是否存在五子连线，如存在则返回True
                            text = ''
                            if gobang.black_turn:  # check_at会切换落子的顺序，所以轮到黑方落子，意味着最后落子方是白方，所以白方顺利
                                text = 'White wins.'
                            else:
                                text = 'Black wins.'
                            gameover_text = font.render(text, True, (255, 0, 0))
                            gobang.chessboard().blit(gameover_text, (round(width / 2 - gameover_text.get_width() / 2),
                                                                    round(height / 2 - gameover_text.get_height() / 2)))
                            game_over = True
                    else:
                        print('此位置已占用，不能在此落子')

        screen.blit(gobang.chessboard(), (0, 0))
        pygame.display.update()
    pygame.quit()