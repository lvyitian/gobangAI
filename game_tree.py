from data_structure import *
from network_evaluation import evaluator


def evaluate_one(cbmap, nag=-1):
    count_value = [0, 1, 10, 100, 10000, 1e7, -1e7, -100000, -1000, -10, -1]
    value = 0
    # 行
    for i in range(15):
        start = 0
        count = 0
        for end in range(15):
            if cbmap[i][end] == nag:  # 遇到黑棋
                start = end + 1
                count = 0
            else:
                count += cbmap[i][end]
                dis = end - start
                if dis > 4:
                    count -= cbmap[i][start]
                    start += 1
                    dis -= 1
                if dis == 4:
                    value += count_value[count]

    # 列
    for i in range(15):
        start = 0
        count = 0
        for end in range(15):
            if cbmap[end][i] == nag:  # 遇到黑棋
                start = end + 1
                count = 0
            else:
                count += cbmap[end][i]
                dis = end - start
                if dis > 4:
                    count -= cbmap[start][i]
                    start += 1
                    dis -= 1
                if dis == 4:
                    value += count_value[count]
    # 斜↘
    for i in range(11):
        start = 0
        count = 0
        for end in range(15 - i):
            if cbmap[i + end][end] == nag:  # 遇到黑棋
                start = end + 1
                count = 0
            else:
                count += cbmap[i + end][end]
                dis = end - start
                if dis > 4:
                    count -= cbmap[i + start][start]
                    start += 1
                    dis -= 1
                if dis == 4:
                    value += count_value[count]
    for i in range(1, 11):
        start = 0
        count = 0
        for end in range(15 - i):
            if cbmap[end][i + end] == nag:  # 遇到黑棋
                start = end + 1
                count = 0
            else:
                count += cbmap[end][i + end]
                dis = end - start
                if dis > 4:
                    count -= cbmap[start][i + start]
                    start += 1
                    dis -= 1
                if dis == 4:
                    value += count_value[count]

    # 斜↗
    for i in range(4, 15):
        start = 0
        count = 0
        for end in range(i + 1):
            if cbmap[i - end][end] == nag:  # 遇到黑棋
                start = end + 1
                count = 0
            else:
                count += cbmap[i - end][end]
                dis = end - start
                if dis > 4:
                    count -= cbmap[i - start][start]
                    start += 1
                    dis -= 1
                if dis == 4:
                    value += count_value[count]
    for i in range(1, 11):
        start = 0
        count = 0
        for end in range(15 - i):
            if cbmap[14 - end][i + end] == nag:  # 遇到黑棋
                start = end + 1
                count = 0
            else:
                count += cbmap[14 - end][i + end]
                dis = end - start
                if dis > 4:
                    count -= cbmap[14 - start][i + start]
                    start += 1
                    dis -= 1
                if dis == 4:
                    value += count_value[count]
    return value


# def evaluate(cbmap):
#     return evaluate_one(cbmap) + evaluate_one(cbmap, 1)

def evaluate(cbmap):
    if not hasattr(evaluate, 'eva'):
        evaluate.eva = evaluator()
    return evaluate.eva.evaluate(cbmap)

def create_cbmap(coords=None):
    cbmap = [[0 for i in range(15)] for i in range(15)]
    if coords is None:
        return cbmap
    for coord in coords:
        x, y = coord['coord']
        cbmap[x][y] = 1 if coord['type'] == 1 else -1
    return cbmap


def check_neighbor(cbmap, i, j, dis):
    pos = []
    if dis == 1:
        pos = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
    if dis == 2:
        pos = [(2, 2), (2, 1), (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2),
               (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 2)]
    for x, y in pos:
        pos_x = i + x
        pos_y = j + y
        if 0 <= pos_x <= 14 and 0 <= pos_y <=14 and (cbmap[pos_x][pos_y] == 1 or cbmap[pos_x][pos_y] == -1):
            return True
    return False


def get_drop_list(cbmap):
    drops = []
    drops_far = []
    for i in range(15):
        for j in range(15):
            if cbmap[i][j] == 0:
                if check_neighbor(cbmap, i, j, 1):
                    drops.append({"x": i, "y": j})
                elif check_neighbor(cbmap, i, j, 2):
                    drops_far.append({"x": i, "y": j})
    drops.extend(drops_far)
    return drops


def min_search(cbmap, depth, alpha, beta):
    if depth == 0:
        return evaluate(cbmap), 0, 0
    best = 1e10
    x, y = 0, 0

    drops = get_drop_list(cbmap)
    for drop in drops:
        i = drop['x']
        j = drop['y']
        cbmap[i][j] = -1
        value, a, b = max_search(cbmap, depth - 1, best if best < alpha else alpha, beta)
        cbmap[i][j] = 0
        if value < best:
            best = value
            x, y = i, j
        if value < beta:
            break

    return best, x, y


def max_search(cbmap, depth, alpha, beta):
    if depth == 0:
        return evaluate(cbmap), 0, 0
    best = -1e10
    x, y = 0, 0

    drops = get_drop_list(cbmap)
    for drop in drops:
        i = drop['x']
        j = drop['y']
        cbmap[i][j] = 1
        value, a, b = min_search(cbmap, depth - 1, alpha, best if best > beta else beta)
        cbmap[i][j] = 0
        if value > best:
            best = value
            x, y = i, j
        if value > alpha:
            break

    return best, x, y


def main():
    return 0


main()
