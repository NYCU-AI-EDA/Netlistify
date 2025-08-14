# %%
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, box


# %%
class seg_fig:
    def __init__(self, path, img_name, size, plot, draw=False, draw_nodes=False):
        self.path = path
        self.img_name = img_name
        self.size = size
        self.draw = draw
        self.plot = plot
        self.draw_nodes = draw_nodes
        self.img = cv2.imread(self.path + "images/" + self.img_name + ".jpg")
        self.nodes = pickle.load(open(self.path + "pkl/" + self.img_name + ".pkl", "rb"))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.width, self.height = self.img.shape[1], self.img.shape[0]
        self.mask = np.ones((self.height, self.width), dtype=np.uint8) * 255
        self.mask = self.get_box()
        self.img = np.concatenate((self.img, self.mask[:, :, np.newaxis]), axis=2)
        # self.img = self.get_node()
        self.img_pad = self.img.copy()
        self.img_pad = self.padding()
        self.width_pad, self.height_pad = self.img_pad.shape[1], self.img_pad.shape[0]
        self.img_pad = self.get_node(self.img_pad, self.draw_nodes)
        # self.small_images = self.get_small_images()
        self.gen_small_images()

    def get_box(self):
        txt = open(self.path + "txt/" + self.img_name + ".txt", "r")
        lines = txt.readlines()
        for line in lines:
            line = line.split()
            cls = int(line[0])
            x_yolo = float(line[1])
            y_yolo = float(line[2])
            yolo_width = float(line[3])
            yolo_height = float(line[4])

            # Convert Yolo Format to Pascal VOC format
            box_width = yolo_width * self.width
            box_height = yolo_height * self.height
            x_min = str(int(x_yolo * self.width - (box_width / 2)))
            y_min = str(int(y_yolo * self.height - (box_height / 2)))
            x_max = str(int(x_yolo * self.width + (box_width / 2)))
            y_max = str(int(y_yolo * self.height + (box_height / 2)))
            # self.mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 127
            self.mask[int(y_min) : int(y_max), int(x_min) : int(x_max)] = cls * 12
        return self.mask

    def get_node(self, img, draw_nodes=False):
        net = []
        for name, node in self.nodes.items():
            for i in range(len(node)):
                # print(name, node[i])
                node1 = int(node[i][0] * self.width), int(node[i][1] * self.height)
                node2 = int(node[i][2] * self.width), int(node[i][3] * self.height)
                if self.draw_nodes:
                    node1_cv = (node1[0], self.height_pad - node1[1])
                    node2_cv = (node2[0], self.height_pad - node2[1])
                    cv2.circle(img, node1_cv, 5, (0, 255, 0), -1)
                    cv2.circle(img, node2_cv, 5, (0, 255, 0), -1)
        if self.draw_nodes:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # print(img.shape)
            return img
        else:
            return img

    def padding(self):
        h, w, c = self.img.shape
        if h % self.size != 0:
            pad_h = self.size - (h % self.size)
            self.img_pad = np.pad(
                self.img_pad, ((pad_h, 0), (0, 0), (0, 0)), mode="constant", constant_values=255
            )
        if w % self.size != 0:
            pad_w = self.size - (w % self.size)
            self.img_pad = np.pad(
                self.img_pad, ((0, 0), (0, pad_w), (0, 0)), mode="constant", constant_values=255
            )
        return self.img_pad

    def shift(self, x, id):
        value = x - self.size * id
        return value

    def small_get_net(self, netlist_pad, w_id, h_id, small, color=(0, 255, 0)):
        for net in netlist_pad:
            if len(net) == 1:
                p1 = (self.shift(net[0][0], w_id), self.shift(net[0][1], h_id))
                netlist_pad[netlist_pad.index(net)] = [p1]
                if self.draw:
                    cv2.circle(small, (p1[0], self.size - p1[1]), 3, color, -1)
            else:
                # print('net',net, len(net))
                p1 = (self.shift(net[0][0], w_id), self.shift(net[0][1], h_id))
                # print('p1',p1)
                p2 = (self.shift(net[1][0], w_id), self.shift(net[1][1], h_id))
                netlist_pad[netlist_pad.index(net)] = [p1, p2]

                # print(p1)
                # print(p2)
                if self.draw:
                    cv2.circle(small, (p1[0], self.size - p1[1]), 3, color, -1)
                    cv2.circle(small, (p2[0], self.size - p2[1]), 3, color, -1)
        # netlist_pad = list(np.array(netlist_pad).reshape(-1, 4))
        return netlist_pad, small

    def gen_small_images(self):
        def tidyup_line(prop):
            # prop is the list of line segments, such as [[a,b,c,d],[...]], (a,b) is the start point, (c,d) is the end point
            def find_ele(arr, ele):
                ind = (arr == ele).all(axis=2).all(axis=1)
                return np.where(ind)

            prop = np.array(prop).round(4).reshape(-1, 2, 2)
            for p in prop:
                if p[0, 0] != p[1, 0] and p[0, 1] != p[1, 1]:
                    prop = np.delete(prop, find_ele(prop, p), 0)
                    left_down = True
                    try:
                        try:
                            shift_point = [p[0, 0], p[1, 1]]
                            shift_point_ind = np.where(
                                (prop == shift_point).all(axis=-1).any(axis=1)
                            )[0][0]
                        except:
                            left_down = False
                            shift_point = [p[0, 0], p[0, 1]]
                            shift_point_ind = np.where(
                                (prop == shift_point).all(axis=-1).any(axis=1)
                            )[0][0]
                        if np.array_equal(prop[shift_point_ind][0], shift_point):
                            prop[shift_point_ind][0][0] = p[1][0]
                        else:
                            prop[shift_point_ind][1][0] = p[1][0]
                    except:
                        pass
                    arr = []
                    for x in prop:
                        if (
                            x[0, 0] == p[1, 0]
                            and x[1, 0] == p[1, 0]
                            and (
                                x[0, 1] == p[0 if left_down else 1, 1]
                                or x[1, 1] == p[0 if left_down else 1, 1]
                            )
                        ):
                            arr.append(True)
                        else:
                            arr.append(False)
                    if sum(arr) == 2:
                        cand = sorted(prop[arr], key=lambda x: x[0][1])
                        prop = prop[np.logical_not(arr)]
                        cand[0][1][1] = cand[1][1][1]
                        prop = np.vstack([prop, [cand[0]]])

            return prop

        def box_filter(netlist, rec):
            netlist_w_pad = []
            for net in netlist:
                # print(net)
                line = LineString(net)
                intersection = line.intersection(rec)
                if intersection:
                    # coord = list(zip(*intersection.xy))
                    coord = [(int(x), int(y)) for x, y in zip(*intersection.xy)]
                    # print('intersect',coord)
                    netlist_w_pad.append(coord)
            return netlist_w_pad

        true_net_dict = {}
        netlist_pad_shift = {}
        netlist_pad = {}

        total_h_idx = int(self.height_pad / self.size)
        total_w_idx = int(self.width_pad / self.size)
        # print('count',total_h_idx* total_w_idx)
        # h_id = 0
        # w_id = 3
        figures = [[None] * total_w_idx for _ in range(total_h_idx)]

        for name, node in self.nodes.items():
            self.nodes[name] = tidyup_line(node)

        for name, node in self.nodes.items():
            net = []
            # print(node)
            node = np.array(node).reshape(-1, 4)
            for i in range(len(node)):
                node1 = int(node[i][0] * self.width), int(node[i][1] * self.height)
                node2 = int(node[i][2] * self.width), int(node[i][3] * self.height)
                net.append([node1, node2])

            true_net_dict[name] = net
            # print(name)
        # print(netlist[2])

        for h_id in range(total_h_idx):
            for w_id in range(total_w_idx):
                # print('h_id:', h_id, 'w_id:', w_id)
                # rec = box(70,0, 139, 69)# (minx, miny, maxx, maxy)
                rec = box(
                    self.size * w_id,
                    self.size * h_id,
                    self.size * (w_id + 1),
                    self.size * (h_id + 1),
                )
                # print(rec)

                # netlist_pad = box_filter(netlist[4], rec)
                for name, nets in true_net_dict.items():
                    # print(name, nets)
                    netlist_pad[name] = box_filter(nets, rec)

                # netlist_pad = box_filter(true_net_dict['net0'], rec)

                # print(netlist_pad)

                # small = seg.img_pad[280:350, 70:140, :] #miny:maxy, minx:maxx
                maxy = self.height_pad - self.size * h_id
                miny = self.height_pad - self.size * (h_id + 1)
                # print(miny, maxy, 70*w_id, 70*(w_id+1))
                small = self.img_pad[miny:maxy, self.size * w_id : self.size * (w_id + 1), :]
                # print(small.shape)
                if self.draw:
                    small = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

                # colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
                colors = [
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 0, 0),
                    (0, 255, 255),
                    (255, 255, 0),
                    (255, 0, 255),
                    (128, 128, 128),
                    (128, 0, 0),
                    (0, 128, 0),
                    (0, 0, 128),
                    (128, 128, 0),
                    (128, 0, 128),
                    (0, 128, 128),
                ]
                for name, nets in netlist_pad.items():
                    used_colors = set()
                    if self.draw:
                        available_colors = [color for color in colors if color not in used_colors]
                        if available_colors:
                            color = available_colors[np.random.randint(0, len(available_colors))]
                            used_colors.add(color)
                    else:
                        color = (0, 0, 0)
                    # print(color)
                    netlist_pad_shift[name], small = self.small_get_net(
                        nets, w_id, h_id, small, color=color
                    )
                    # print(name, netlist_pad_shift[name])

                ###############存小圖的pkl路徑#########################
                # with open(f'./data_size_50/pkl/{self.img_name}_{h_id}_{w_id}.pkl', 'wb') as f:
                #     pickle.dump(netlist_pad_shift, f)

                # print(small.shape)
                ###############存小圖的路徑#########################
                # cv2.imwrite(f'./data_size_50/images/{self.img_name}_{h_id}_{w_id}.png', small)

                figures[h_id][w_id] = small
                # plt.figure(figsize=(5, 5))
                # plt.imshow(small)
                # plt.axis('off')

        if self.plot:
            fig, axs = plt.subplots(total_h_idx, total_w_idx)
            fig.patch.set_facecolor("lightgrey")
            # print(axs)
            # plt.figsize=(30, 20)
            for i in range(total_h_idx):
                for j in range(total_w_idx):
                    axs[total_h_idx - i - 1, j].imshow(figures[i][j])
                    axs[total_h_idx - i - 1, j].axis("off")
            plt.figsize = (50, 50)
            # plt.savefig(f'./test.png')
            plt.show()


# %%
data_dir = "../gen_sp/data/"
files = os.listdir(data_dir + "images/")
# print(files[:5])
# for file in files[:2000]:
#     seg_fig(data_dir, file.split('.')[0],50,False , False, False)
seg_fig(data_dir, "circuit9", size=50, plot=True, draw=True, draw_nodes=False)

# %%
# cv2.imwrite('./data/images'+file.split('.')[0]+'.jpg', img)
# with open(f'../gen_sp/data/pkl/{file.split(".")[0]}.pkl', 'wb') as f:
#             pickle.dump(dict_net_scale, f)
# img_test = cv2.imread('./data/images/circuit43175_1_1.png', cv2.IMREAD_UNCHANGED)
# print(img_test.shape)
# plt.imshow(img_test)
# plt.axis('off')
# plt.show()

# nets = pickle.load(open('./data/pkl/circuit43175_1_1.pkl', 'rb'))
# print(nets)
files = os.listdir("./data_size_50/images/")
print(len(files))
# %%
# print(os.getcwd())
# os.chdir('/home/111/hank/segment_pic')
# print(os.getcwd())
