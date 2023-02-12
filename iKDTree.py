import numpy as np
from math import sqrt

"""
KD树，k-Dimention tree.用来组织数据的树形数据结构，以便快速的查找某些元素。
用来实现KNN算法中查找m个与目标点最近的样本点。在样本点数量远远大于样本点维度时，KD树(O(logN))查找时间复杂度要比线性搜索(O(N^2)低不少。
"""


class KDNode(object):
    def __init__(self, value, split, left, right):
        # value=[x,y]
        self.value = value
        self.split = split  # 分叉的地方是第几维度
        self.right = right
        self.left = left


class KDTree(object):
    def __init__(self, data):
        # data=[[x1,y1],[x2,y2]...,] 只能是二维列表，若要支持ndarray需要改写一下。
        # 维度
        k = len(data[0])

        def CreateNode(split, data_set):
            if not data_set:
                return None
            data_set.sort(key=lambda x: x[split])
            # 整除2
            split_pos = len(data_set) // 2
            median = data_set[split_pos]
            split_next = (split + 1) % k

            return KDNode(median, split, CreateNode(split_next, data_set[: split_pos]),
                          CreateNode(split_next, data_set[split_pos + 1:]))

        self.root = CreateNode(0, data)

    def search(self, root: KDNode, x, count=1):
        """
        :param root:
        :param x:
        :param count:
        :return:
        """
        nearest = []
        for i in range(count):
            nearest.append([-1, None])
        self.nearest = np.array(nearest)

        def recurve(node):
            if node is not None:
                axis = node.split
                daxis = x[axis] - node.value[axis]
                if daxis < 0:
                    recurve(node.left)
                else:
                    recurve(node.right)
                dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(x, node.value)))  # 目标点和最近点的欧式距离
                for i, d in enumerate(self.nearest):
                    if d[0] < 0 or dist < d[0]:  # 如果当前nearest内i处未标记（-1），或者新点与x距离更近
                        self.nearest = np.insert(self.nearest, i, [dist, node.value], axis=0)  # 插入比i处距离更小的
                        self.nearest = self.nearest[:-1]
                        break
                # 找到nearest集合里距离最大值的位置，为-1值的个数
                n = list(self.nearest[:, 0]).count(-1)  # list.count()函数用来统计列表中某个值元素出现的次数
                # 切分轴的距离比nearest中最大的小（存在相交）
                if self.nearest[-n - 1, 0] > abs(daxis):
                    if daxis < 0:  # 相交，x[axis]< node.data[axis]时，去右边（左边已经遍历了）
                        recurve(node.right)
                    else:  # x[axis]> node.data[axis]时，去左边，（右边已经遍历了）
                        recurve(node.left)

        recurve(root)
        return self.nearest


data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
kd = KDTree(data)

# [3, 4.5]最近的3个点
n = kd.search(kd.root, [3, 4.5], 3)
print(n)

# [[1.8027756377319946 list([2, 3])]
# [2.0615528128088303 list([5, 4])]
# [2.692582403567252 list([4, 7])]]

# python新内容小结:
# 1、enumerate(list)，同时遍历索引和内容
# for i , data in enumerate(list):
#     print(i,data)
# 2、 zip(x,y),遍历时取x的一个元素和y的一个元素来进行。
# 3、list按照指定关键字排序
# list.sort(key=lambda x: x[0])
