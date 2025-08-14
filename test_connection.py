import json
import math
import os
from collections import Counter


# 計算兩點之間距離的輔助函數
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 用來尋找每個線段所屬的組的根節點
def find(parent, i):
    if parent[i] == i:
        return i
    else:
        return find(parent, parent[i])


# 用來合併兩個組
def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)

    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1


# 根據兩條線段端點的距離將它們分組
def group_segments(segments, threshold=0.2):
    n = len(segments)
    parent = list(range(n))
    rank = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            # 比較線段 i 和線段 j 之間的所有端點組合
            if (
                distance(segments[i][0], segments[j][0]) < threshold
                or distance(segments[i][0], segments[j][1]) < threshold
                or distance(segments[i][1], segments[j][0]) < threshold
                or distance(segments[i][1], segments[j][1]) < threshold
            ):
                # 如果有任一對端點距離小於閾值，則將這兩個線段分為同一組
                union(parent, rank, i, j)

    # 根據 parent 結構將線段分組
    groups = {}
    for i in range(n):
        root = find(parent, i)
        if root not in groups:
            groups[root] = []
        groups[root].append(segments[i])

    return list(groups.values())


# 覆蓋原資料，使相同的點合併為同一點
def merge_and_replace_points(points, threshold):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if distance(points[i], points[j]) < threshold:
                # 將點 j 的座標覆蓋為點 i 的座標
                points[j] = points[i]
    return points


# 此處修改路徑
with open("line_sg_test/circuit444_2_10.json", "r") as file:
    data = json.load(file)
    data = [
        [[0.28911569714546204, 0.8890003561973572], [0.36341023445129395, 0.8891093134880066]],
        [[0.36442211270332336, 0.8891484141349792], [0.4553734064102173, 0.8891320824623108]],
        [[0.4555714428424835, 0.8893438577651978], [0.4661135971546173, 0.8893664479255676]],
        [[0.4643342196941376, 0.8571428656578064], [0.4643479287624359, 0.8890666961669922]],
        [[0.4640902578830719, 0.7857142686843872], [0.4639827013015747, 0.841768205165863]],
        [[0.4641771912574768, 0.8421052098274231], [0.5461963415145874, 0.842009425163269]],
        [[0.46364736557006836, 0.8414470553398132], [0.4634774625301361, 0.8569403290748596]],
        [[0.5466442108154297, 0.8449260592460632], [0.6375227570533752, 0.8448331356048584]],
        [[0.28908440470695496, 0.7142857313156128], [0.28938156366348267, 0.7556429505348206]],
        [[0.46490615606307983, 0.7143615484237671], [0.46482157707214355, 0.785619854927063]],
        [[0.28953835368156433, 0.6430600881576538], [0.2895766496658325, 0.7142857313156128]],
        [[0.46490615606307983, 0.6429329514503479], [0.46482157707214355, 0.7141912579536438]],
        [[0.6392593383789062, 0.6461939811706543], [0.6394219398498535, 0.7042826414108276]],
        [[0.28953543305397034, 0.5716264247894287], [0.2895732820034027, 0.6428571343421936]],
        [[0.4657800495624542, 0.6105548143386841], [0.465933620929718, 0.6428571343421936]],
        [[0.6409390568733215, 0.5714285969734192], [0.6412264704704285, 0.6428571343421936]],
        [[0.6409390568733215, 0.5], [0.6412264704704285, 0.5714285969734192]],
        [[0.4794604480266571, 0.47753235697746277], [0.5464481115341187, 0.4776766896247864]],
        [[0.5464481115341187, 0.4763253331184387], [0.6375227570533752, 0.4763502776622772]],
        [[0.6379201412200928, 0.4776424169540405], [0.641010582447052, 0.4775785207748413]],
        [[0.6418352723121643, 0.47687608003616333], [0.64201420545578, 0.4999217092990875]],
        [[0.28953835368156433, 0.3573457896709442], [0.2895766496658325, 0.4285714328289032]],
        [[0.28880539536476135, 0.3373992145061493], [0.28911739587783813, 0.3571428656578064]],
        [[0.2886195778846741, 0.14300549030303955], [0.28913480043411255, 0.20050384104251862]],
        [[0.28816911578178406, 0.11210856586694717], [0.28852468729019165, 0.1428571492433548]],
    ]
    # data.append(data[0])
# 使用列表生成式將最內層的列表轉換為 tuple
converted_data = [[tuple(inner_list) for inner_list in outer_list] for outer_list in data]

# 輸出結果
# print(converted_data)


# 根據距離將線段分組
grouped_segments = group_segments(converted_data, threshold=0.08)  # 閾值根據需求調整
# print(grouped_segments)
group_connection = []

for i in range(len(grouped_segments)):
    # 將所有點展平為一個列表
    all_points = [point for pair in grouped_segments[i] for point in pair]

    # 計算每個點的出現次數
    updated_points = merge_and_replace_points(all_points, threshold=0.001)
    # print(all_points)
    # print(updated_points)
    # exit()
    # print(merged_points)
    point_count = Counter(updated_points)

    # 只保留出現一次的點
    unique_points = [point for point in updated_points if point_count[point] == 1]
    group_connection.append(unique_points)
from pprint import pprint

print(group_connection)
