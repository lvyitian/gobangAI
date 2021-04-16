from collections import namedtuple
import pickle

Position = namedtuple('Position', ['x', 'y'])


class data_sl():
    @staticmethod
    def save(data, path):
        data = pickle.dumps(data)
        f = open(path, 'wb')
        f.write(data)
        f.close()

    @staticmethod
    def load(path):
        f = open(path, 'rb')
        data = f.read()
        data = pickle.loads(data)
        f.close()
        return data
