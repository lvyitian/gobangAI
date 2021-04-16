import sys

from make_trains import *
from data_structure import *
from game import game


def calc_entropy(dataset):
    if len(dataset) == 0:
        return 0.0
    from math import log2
    count_black = 0
    count_white = 0
    for i in dataset:
        if i[1] == -1:
            count_black += 1
        elif i[1] == 1:
            count_white += 1
    prob = [0, 0, 0]
    prob[0] = count_white / len(dataset)
    prob[1] = count_black / len(dataset)
    prob[2] = (len(dataset) - count_black - count_white) / len(dataset)
    entropy = 0.0
    for i in range(3):
        if prob[i] != 0 :
            entropy -= prob[i] * log2(prob[i])
    return entropy


def find_best(dataset):
    dataset = dataset[:]
    keys = (list)(dataset[0].keys())
    datalen = len(dataset)
    min_ent = 1.0
    value = 0
    field = keys[0]
    all_ent = 1.0
    for i in range(3):
        dataset = sorted(dataset, key=lambda x:x[keys[i]])
        data = []
        for j in range(datalen):
            data.append([dataset[j][keys[i]], dataset[j][keys[3]]])

        all_ent = calc_entropy(data)
        for j in range(datalen - 1):
            less_data, more_data = cut_dataset(data, 0, data[j][0])
            less_ent = calc_entropy(less_data)
            more_ent = calc_entropy(more_data)
            entropy = j / datalen * less_ent + (datalen - j) / datalen * more_ent
            if entropy < min_ent:
                min_ent = entropy
                value = data[j][0]
                field = keys[i]
    if min_ent > all_ent:
        value = 0.0
    return field, value


def cut_dataset(dataset, field, value):
    less_data = []
    more_data = []
    for data in dataset:
        if data[field] <= value:
            less_data.append(data)
        else:
            more_data.append(data)
    return less_data, more_data


def create_tree(dataset):
    field, value = find_best(dataset)
    tree = {'field': field, 'value': value}
    less_data, more_data = cut_dataset(dataset, field, value)
    if len(less_data) == 0 or len(more_data) == 0:
        tree['son'] = dataset[0]['type']
    else:
        tree['son'] = {'less': create_tree(less_data), "more": create_tree(more_data)}
    return tree


def test_tree(tree, dataset):
    global ret
    if type(tree['son']).__name__ == 'dict':
        if dataset[tree['field']] <= tree['value']:
            ret = test_tree(tree['son']['less'], dataset)
        else:
            ret = test_tree(tree['son']['more'], dataset)
    else:
        ret = tree['son']
    return ret


def train_data(num):
    print("making training data...")
    dataset = []
    for i in range(num):
        impath = "out\\image" + str(i) + ".jpg"
        dtpath = "out\\coord" + str(i) + ".dat"
        src = cv2.imread(impath)
        coords = data_sl.load(dtpath)
        dataset.extend(make_trains(src, coords))
        print("data added for image" + str(i))
    print("done.")

    print("constructing decision tree...")
    tree = create_tree(dataset)
    print("done.")

    data_sl.save(tree, "out\\tree.dat")


def test_data(src):
    print("making testing data...")
    testdata = make_tests(src)
    print("done.")

    print("testing")
    tree = data_sl.load("out\\tree.dat")
    coords = []
    for i in range(len(testdata)):
        color = test_tree(tree, testdata[i])
        if color != 0:
            coords.append({"type": 1 if color == 1 else 0, "coord": Position(i % 15, i // 15)})
    print("done.")
    game(coords)


def main():
    sys.setrecursionlimit(3000)
    train_data(3)
    # src = cv2.imread("test\\detect\\image3.jpg")
    # test_data(src)

    return 0


main()
