"""
使用scipy实现的KDTree接口。
"""
import numpy as np
from scipy.spatial import KDTree

if __name__ == '__main__':
    data_set = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    kd_tree = KDTree(data_set)
    d, i = kd_tree.query(np.array([3, 3.5]), k=3)  # 距离点(3,3.5)最近的三个点
    print("data_set:", data_set, sep='\n')
    print("nearest distance:", d, sep='\n')
    print("nearest point's index:", i, sep='\n')
