# %%
import itertools
import os
import sys
from collections import Counter, defaultdict
from itertools import chain

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# 計算兩點之間距離的輔助函數
def build_connection(data, distance, similar_threshold, threshold, duplicate_threshold):
    # 去除相似的線段
    def remove_similar_segments(segment, threshold):
        unique_segments = []
        for i in range(len(segment)):
            for j in range(i + 1, len(segment)):
                if distance(segment[i], segment[j]) <= threshold:
                    break
            else:
                unique_segments.append(segment[i])
        return unique_segments

    # 用來尋找每個線段所屬的組的根節點
    def find(parent, i):
        return i if parent[i] == i else find(parent, parent[i])

    # 根據兩條線段端點的距離將它們分組
    def group_segments(segments, similar_threshold, threshold=0.2):
        # 去除相似線段
        unique_segments = [
            remove_similar_segments(segment, similar_threshold) for segment in segments
        ]
        n = len(unique_segments)

        parent = list(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                # 比較線段 i 和線段 j 之間的所有端點組合
                for point1, point2 in itertools.product(unique_segments[i], unique_segments[j]):
                    if distance(point1, point2) <= threshold:
                        # print(f"merge {unique_segments[i]} and {unique_segments[j]}")
                        rootX = find(parent, i)
                        rootY = find(parent, j)
                        parent[rootY] = rootX
                        break
        # 根據 parent 結構將線段分組
        groups = [[] for _ in range(n)]
        for i in range(n):
            root = find(parent, i)
            groups[root].append(unique_segments[i])
        groups = [group for group in groups if len(group) > 0]

        return groups

    def set_of_points(data):
        s = defaultdict(set)
        for i, d in enumerate(data):
            for dd in d:
                s[tuple(dd)].add(i)
        return s

    # 覆蓋原資料，使相同的點合併為同一點
    def merge_and_replace_points(points, threshold, points_group):
        merged_points = points[:]
        for i in range(len(merged_points)):
            for j in range(i + 1, len(merged_points)):
                if distance(merged_points[i], merged_points[j]) <= threshold:
                    group1 = points_group[tuple(merged_points[i])]
                    group2 = points_group[tuple(merged_points[j])]
                    if group1.intersection(group2):
                        continue
                    # 將點 j 的座標覆蓋為點 i 的座標
                    # print(f"merge {merged_points[j]} to {merged_points[i]}")
                    merged_points[j] = merged_points[i]
        return merged_points

    points_group = set_of_points(data)
    grouped_segments = group_segments(data, similar_threshold, threshold)  # 閾值根據需求調整

    group_connection = []

    for grouped_segment in grouped_segments:
        # 將所有點展平為一個列表

        all_points = list(chain.from_iterable(grouped_segment))
        # 計算每個點的出現次數
        updated_points = merge_and_replace_points(all_points, duplicate_threshold, points_group)
        updated_points = list(map(tuple, updated_points))
        point_count = Counter(updated_points)

        # 只保留出現一次的點
        unique_points = [point for point in updated_points if point_count[point] == 1]
        if len(unique_points) > 0:
            group_connection.append(unique_points)
    return group_connection


def build_connection_v2(data, distance, threshold):

    # 用來尋找每個線段所屬的組的根節點
    def find(parent, i):
        return i if parent[i] == i else find(parent, parent[i])

    # 根據兩條線段端點的距離將它們分組
    def group_segments(segments, threshold=0.2):
        n = len(segments)

        parent = list(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                # 比較線段 i 和線段 j 之間的所有端點組合
                for point1, point2 in itertools.product(segments[i], segments[j]):
                    if distance(point1, point2) <= threshold:
                        rootX = find(parent, i)
                        rootY = find(parent, j)
                        parent[rootY] = rootX
                        break
        # 根據 parent 結構將線段分組
        groups = [[] for _ in range(n)]
        for i in range(n):
            root = find(parent, i)
            groups[root].append(segments[i])
        groups = [group for group in groups if len(group) > 0]

        return groups

    grouped_segments = group_segments(data, threshold)  # 閾值根據需求調整

    return grouped_segments


if __name__ == "__main__":
    if not os.getcwd().endswith("ViT-Schematic"):
        os.chdir("..")
    else:
        sys.path.append(".")
    import json

    from Model import *

    # 此處修改路徑
    data = [[[0.14, 0.52], [0.74, 0.54]], [[0.74, 0.55], [0.74, 0.23]]]
    data = [
        [[0.74, 0.12], [0.74, 0.54], [0.74, 0.53], [0.74, 0.94]],
        [[0.74, 0.55], [0.1, 0.55], [0.9, 0.55]],
    ]
    data = [
        [[0.28908440470695496, 0.7142857313156128], [0.28938156366348267, 0.7556429505348206]],
        [[0.28953835368156433, 0.6430600881576538], [0.2895766496658325, 0.7142857313156128]],
    ]
    data = [
        [[0.28953835368156433, 0.3573457896709442], [0.2895766496658325, 0.4285714328289032]],
        [[0.28880539536476135, 0.3373992145061493], [0.28911739587783813, 0.3571428656578064]],
    ]
    data = [
        [[0.05310596525669098, 0.44958990812301636], [0.0714285746216774, 0.6]],
        [
            [0.07296573370695114, 0.6],
            [0.08410254865884781, 0.9],
        ],
    ]
    data = [
        [[0.09524867683649063, 0.6234647631645203], [0.1428571492433548, 0.6221328973770142]],
        [[0.1567024141550064, 0.5360572338104248], [0.2142857164144516, 0.5360572338104248]],
        [[0.14573770761489868, 0.5157356262207031], [0.1428571492433548, 0.6221328973770142]],
    ]
    data = [
        [[0.80000001, 0.55000003], [1.0, 0.53749996]],
        [[0.81874999, 0.52499998], [0.81874999, 1.0]],
    ]
    # data = [
    #     [[0.01, 0.01], [0.74, 0.53]],
    #     [[0.0, 0.001], [0.0, 0.02]],
    #     [[0.0, 0.0], [0.0, 0.0]],
    # ]
    # pprint(data)
    group_connection = build_connection(
        data, norm1, similar_threshold=0, threshold=0.01, duplicate_threshold=0.02
    )
    img = np.full((512, 512, 3), 255, np.uint8)
    for i, d in enumerate(data):
        color = color_map(i)
        img = draw_point(img, d, color=color, width=8)
    plot_images(img, img_width=300)
    img = np.full((512, 512, 3), 255, np.uint8)
    for i, group in enumerate(group_connection):
        color = color_map(i)
        img = draw_rect(img, group, color=color, width=12)
    # print(group_connection)
    # print(len(group_connection))
    plot_images(img, img_width=300)
    print(group_connection)
