# %%
from collect_connection import build_connection, build_connection_v2

# from main_line import *
from visualizer import get_local

get_local.activate()
from main import *
from Model import *


class TestWindowed(Datasetbehaviour):

    def __init__(self, img, stride, window_size):
        if window_size != -1:
            self.dataset = slice_image_into_windows(img, window_size, stride)
        else:
            self.dataset = [img]
        super().__init__(len(self.dataset), self.__create, always_reset=True, log2console=False)

    def __create(self, i):
        result = self.dataset[i]
        return result, 0


class OneTimeWrapper(Datasetbehaviour):

    def __init__(self, imgs):
        self.dataset = imgs
        super().__init__(len(imgs), self.__create, always_reset=True, log2console=False)

    def __create(self, i):
        return self.dataset[i][0], 0


model = None


def load_model(
    model_path,
):
    model_path = Path(model_path)
    weight_dir = Path(__file__).parent / "runs/FormalDatasetWindowedLinePair"
    weight_dir.mkdir(exist_ok=True)
    model_path_dir = Path(model_path).parent
    weight_path = weight_dir / model_path_dir.name
    if model_path_dir.exists():
        if weight_path.exists():
            shutil.rmtree(weight_path)
        shutil.copytree(model_path_dir, weight_path)
    global model
    print(weight_path)
    if model is None:
        model = Model(
            xtransform=xtransform,
            log2console=False,
        )
        model.fit(
            create_model(),
            pretrained_path=weight_path / model_path.name,
        )


def predict_mask_imgs(imgs, threshold):
    tmp = OneTimeWrapper(imgs)
    result = model.inference(tmp, verbose=False).cpu().numpy()[:, :, :4]
    result = rearrange(result, "a b (c d) -> a b c d", c=2, d=2)
    # plot_images(draw_line(imgs[2][0], result[2]), 300)
    legalized_lines = []
    for i in range(len(result)):
        t = legalize_line(result[i], threshold)
        legalized_lines.append(t)
    # plot_images(draw_line(imgs[2][0], legalized_lines[2]), 300)
    # exit()
    return legalized_lines


np.set_printoptions(precision=2)


def calculate_line_angle(x1, y1, x2, y2):
    # Calculate the differences
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the angle in radians
    angle_radians = math.atan2(abs(dy), abs(dx))

    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def legalize_line(lines, threshold):
    for j, line in enumerate(lines):
        # clip line to 0 and 1
        line = np.clip(line, 0, 1)
        angle = calculate_line_angle(*line[0], *line[1])
        if abs(angle - 45) > min(angle, 90 - angle):
            if abs(line[0, 0] - line[1, 0]) < abs(line[0, 1] - line[1, 1]):
                m = (line[0, 0] + line[1, 0]) / 2
                line[0, 0] = m
                line[1, 0] = m
                if line[0, 1] < line[1, 1]:
                    line = line[[1, 0]]
            else:
                m = (line[0, 1] + line[1, 1]) / 2
                line[0, 1] = m
                line[1, 1] = m
                if line[0, 0] > line[1, 0]:
                    line = line[[1, 0]]
        lines[j] = line
    new_lines = []
    for line in lines:
        if norm1(line[0], line[1]) > threshold:
            new_lines.append(line)
    new_lines = np.array(new_lines)
    return new_lines


@torch.no_grad()
@hidden_matplotlib_plots
def analyze_connection(
    dataset_config,
    input_img,
    draw_img,
    interval,
    local_threshold,
    global_threshold,
    debug,
    debug_img_width,
    debug_cell,
):
    input_img = input_img.copy()
    draw_img = draw_img.copy()
    load_model(config.get_best_model_path(dataset_config))

    def draw_connection(img, groups, grid, scaled):
        img = img.copy()
        for i, group in enumerate(groups):
            group = np.array(group)
            if not scaled:
                group *= cropped_slice_size
                group[:, 0] /= img.shape[1]
                group[:, 1] /= img.shape[0]
            color = color_map(i)
            img = draw_rect(img, group, color=color, width=5)
        plot_images(
            (
                create_grid(
                    img,
                    window_size=cropped_slice_size,
                    padding=1,
                    pad_value=127,
                )
                if grid
                else img
            ),
            debug_img_width,
        )
        return img

    with HiddenPrints(disable=debug), HiddenPlots(disable=debug):
        slice_size = config.IMAGE_SIZE
        cropped_slice_size = slice_size - 2 * interval
        ori_img = padding(input_img, cropped_slice_size)
        num_column = math.ceil(ori_img.shape[1] / cropped_slice_size)
        num_row = math.ceil(ori_img.shape[0] / cropped_slice_size)
        # print("num_column:", num_column)
        # print("num_row:", num_row)

        image_copy = ori_img.copy()
        ori_img = resize_with_padding(ori_img, interval * 2, interval * 2, fill=255)
        ori_img = shift(ori_img, (interval, interval), fill=255)
        grid_spacing_ratio = interval / slice_size
        r = 0
        second_box = box(
            grid_spacing_ratio + r,
            grid_spacing_ratio + r,
            1 - grid_spacing_ratio - r,
            1 - grid_spacing_ratio - r,
        )
        difference_box = box(0, 0, 1, 1) - second_box
        dataset = TestWindowed(ori_img, -2 * interval, slice_size)
        image_set = []
        local_lines = {}
        min_line_length = 1e-2
        predict_results = predict_mask_imgs(dataset, min_line_length)
        for i in range(len(dataset)):
            image_ori = dataset[i][0]
            image_bk = dataset[i][0][:, :, :3]
            image_bk_gray = cv2.cvtColor(image_bk, cv2.COLOR_BGR2GRAY)
            image = image_bk.copy()
            row_idx = i // (num_column)
            col_idx = i % (num_column)
            anchor = (
                col_idx * cropped_slice_size,
                (num_row - row_idx - 1) * cropped_slice_size,
            )
            if image.mean() < 254.5:
                lines = []
                lines = predict_results[i]
                # remove point that is on white space
                remove_white_point = True
                if remove_white_point:
                    filtered = []
                    for line in lines:
                        if (
                            CartesianImage(image_ori)[line[0]][-1] != 255
                            and CartesianImage(image_ori)[line[1]][-1] != 255
                        ):
                            filtered.append(False)
                        else:
                            for point in line:
                                start = int(point[0] * slice_size)
                                end = int((1 - point[1]) * slice_size)
                                radius = 3
                                radius_pixel = image_bk_gray[
                                    max(end - radius, 0) : min(end + radius, slice_size - 1),
                                    max(start - radius, 0) : min(start + radius, slice_size - 1),
                                ]
                                if radius_pixel.size > 0 and radius_pixel.mean() >= 250:
                                    filtered.append(False)
                                    break
                            else:
                                filtered.append(True)
                    lines = lines[filtered]

                connection = build_connection_v2(
                    lines,
                    norm1,
                    threshold=local_threshold,
                )
                # if connection:
                #     line_image = draw_line(dataset[i][0], lines)
                #     for i, c in enumerate(connection):
                #         color = color_map(i)
                #         line_image = draw_point(line_image, c, color=color)
                #     plot_images(image, 300)
                #     plot_images(line_image, 300)
                #     print(lines)
                #     print(connection)
                #     input()
                # exit()
                cropped_connection = []
                for c in connection:
                    buffer = []
                    for line in c:
                        k = LineString(line) - difference_box
                        if isinstance(k, LineString):
                            cropped_lines = np.array(k.coords)
                            if cropped_lines.size > 0:
                                buffer.append(cropped_lines)
                    if buffer:
                        buffer = np.array(buffer)
                        buffer -= interval / slice_size
                        buffer *= slice_size / cropped_slice_size
                        buffer = np.clip(buffer, 0, 1)
                        cropped_connection.append(buffer)
                # plot_images(image_ori, 300)
                # plot_images(CartesianImage(image_ori)[interval:-interval, interval:-interval], 300)
                # print(cropped_connection)
                # input()
                if len(cropped_connection) > 0:
                    local_lines[(row_idx, col_idx)] = (
                        cropped_connection,
                        anchor,
                        CartesianImage(image_ori)[
                            interval : -interval + slice_size, interval : -interval + slice_size
                        ],
                    )
                # if [row_idx, col_idx] in [
                #     debug_cell,
                #     [debug_cell[0] + 1, debug_cell[1]],
                #     [debug_cell[0], debug_cell[1] + 1],
                # ]:
                #     plot_images(draw_line(dataset[i][0].copy(), lines, thickness=1), 300)
                #     print("mask mean value:", mask.mean())
                #     print("lines")
                #     print(lines)
                #     plot_images(
                #         draw_line(
                #             CartesianImage(dataset[i][0].copy())[
                #                 interval : -interval + slice_size, interval : -interval + slice_size
                #             ],
                #             new_lines,
                #             thickness=1,
                #         ),
                #         300,
                #     )
                #     print("new_lines")
                #     print(new_lines)A
                image = CartesianImage(image)[
                    interval : -interval + slice_size, interval : -interval + slice_size
                ]
                for i, c in enumerate(cropped_connection):
                    color = color_map(i)
                    image = draw_line(image, c, thickness=2, color=color)
                # plot_images(image, 300)
                # print(cropped_connection)
            else:
                image = CartesianImage(image)[
                    interval : -interval + slice_size, interval : -interval + slice_size
                ]
            image_set.append(image)
        print("model prediction figure")
        plot_images(
            create_grid(
                image_set,
                nrow=num_column,
                padding=1,
                pad_value=127,
            ),
            debug_img_width,
        )
        strict_match = False
        # combine lines between grids
        if strict_match:
            strict_match_threshold=0.15
            threshold = strict_match_threshold
            for i, j in itertools.product(range(num_row), range(num_column)):
                if (i, j) not in local_lines:
                    continue
                on_border_sets, _, img = local_lines[(i, j)]
                if (i, j + 1) in local_lines:
                    qualified = []
                    matches_right = []
                    for on_border_set in on_border_sets:
                        for line in on_border_set:
                            a = line[1]
                            if abs(line[0, 0] - line[1, 0]) > abs(line[0, 1] - line[1, 1]):
                                if a[0] >= 1 - threshold:
                                    qualified.append(a)
                    for lines in local_lines[(i, j + 1)][0]:
                        for line in lines:
                            a = line[0]
                            if a[0] <= threshold:
                                matches_right.append(a)
                    if len(qualified) > 0 and len(matches_right) > 0:
                        shift_match = np.array(matches_right)
                        shift_match[:, 0] += 1
                        matches = linear_sum_assignment(distance.cdist(qualified, shift_match))
                        for aidx, midx in zip(*matches):
                            a = qualified[aidx]
                            m = matches_right[midx]
                            a[0] = 1
                            a[1] = m[1]
                            m[0] = 0

                if (i + 1, j) in local_lines:
                    qualified = []
                    for on_border_set in on_border_sets:
                        for line in on_border_set:
                            a = line[1]
                            if abs(line[0, 0] - line[1, 0]) < abs(line[0, 1] - line[1, 1]):
                                if a[1] <= threshold:
                                    qualified.append(a)
                    # grid bottom
                    matches_bottom = []
                    for lines in local_lines[(i + 1, j)][0]:
                        for line in lines:
                            a = line[0]
                            if a[1] >= 1 - threshold:
                                matches_bottom.append(a)
                    if len(qualified) > 0 and len(matches_bottom) > 0:
                        shift_match = np.array(matches_bottom)
                        shift_match[:, 1] += -1
                        matches = linear_sum_assignment(distance.cdist(qualified, shift_match))
                        for aidx, midx in zip(*matches):
                            a = qualified[aidx]
                            m = matches_bottom[midx]
                            a[0] = m[0]
                            a[1] = 0
                            m[1] = 1

                if [i, j] == debug_cell:
                    plot_images(img, 300)
                    print("group")
                    print(on_border_sets)
                    print("matches right")
                    print(matches_right)
                    print("matches bottom")
                    print(matches_bottom)
        grid_lines = []
        for key, value in list(local_lines.items()):
            group = value[0]
            # print(group)
            for lines in group:
                lines[:, :, 0] += key[1]
                lines[:, :, 1] += num_row - key[0] - 1
                grid_lines.append(lines.flatten().reshape(-1, 2).tolist())

        grid_lines_connection = build_connection(
            grid_lines,
            norm1,
            similar_threshold=0,
            threshold=global_threshold,
            duplicate_threshold=1e-5,
        )

        print("connection figure")
        draw_connection(image_copy, grid_lines_connection, True, False)
        global_connection = []
        for group in grid_lines_connection:
            group = np.array(group) * cropped_slice_size
            group[:, 0] /= draw_img.shape[1]
            group[:, 1] /= draw_img.shape[0]
            global_connection.append(group.tolist())

        draw_img = draw_connection(draw_img, global_connection, False, True)
        cache = get_local.cache
        attention_map = cache["CustomTransformerEncoderLayer.forward"][-1]  # (144, 196, 196)
        s = []
        print(len(attention_map))
        for map in attention_map:
            heatmap_data_normalized = cv2.normalize(
                map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            heatmap_data_normalized = heatmap_data_normalized.astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_data_normalized, cv2.COLORMAP_JET)
            s.append(heatmap_colored)
        s = create_grid(s, nrow=num_column, padding=1, pad_value=127)
        plot_images(s, 600)
        return global_connection, draw_img


if __name__ == "__main__":
    path = "test_images/circuit50038.png"
    path = "real_data/images/000002.jpg"
    test_list = list(
        path_like_sort([x.name for x in Path(config.TEST_DATASET_PATH + "/labels").iterdir()])
    )
    # test_list = ["000230.jpg"]
    if config.REAL_DATA:
        data_config = config.DatasetConfig.REAL
    else:
        data_config = config.DatasetConfig.CC
    img_name = Path(config.TEST_DATASET_PATH + "/images/" + test_list[2])
    img_name = img_name.with_suffix(".jpg")
    img = cv2.imread(img_name)
    label_name = img_name.name.replace(".jpg", ".txt") # load YOLO label for masks
    img, processed_img = load_test_data(img, label_name, config.TEST_DATASET_PATH, data_config)

    plot_images(img, 500)
    group_connection, result_img = analyze_connection(
        data_config,
        processed_img, # images with component masks
        img, 
        interval=5, # overlap distance between adjacent cells 
        local_threshold=0.1, # determine whether the wires should be merged into groups based on the distance threshold in the cell. 
        global_threshold=0.02, # determine whether the wires should be merged into groups based on the distance threshold between the adjacent cells. 
        debug=True, # return images with detected wires
        debug_cell=[-1, -1], # return the wire detection results in the cells you choose
        debug_img_width=500,
    )
# %%
