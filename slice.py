# %%
from shapely.geometry import LineString, box

from main_config import *
from utility import *

class_label_cc = dict(
    enumerate(
        [
            "gnd",
            "pmos",
            "nmos",
            "pnp",
            "npn",
            "resistor",
            "capacity",
            "voltage",
            "current",
            "diode",
            "inductor",
            "and",
            "or",
            "xor",
            "not",
            "func",
            "op",
            "tgate",
        ]
    )
)
class_label_real = dict(
    enumerate(
        [
            "gnd",
            "pmos",
            "nmos",
            "pnp",
            "npn",
            "resistor",
            "capacity",
            "voltage",
            "current",
            "text",
            "node",
            "crossing",
        ]
    )
)


def crop_image(img, x, y, w, h):
    return img[y : y + h, x : x + w]


def get_box(img, path, config: DatasetConfig):
    new_img = np.full_like(img, 255)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img[:, :, 0] = img
    height, width = img.shape
    with open(path, "r") as txt:
        lines = txt.readlines()
        for line in lines:
            line = line.split()
            cls = int(line[0])
            x_yolo = float(line[1])
            y_yolo = float(line[2])
            yolo_width = float(line[3])
            yolo_height = float(line[4])

            # Convert Yolo Format to Pascal VOC format
            box_width = yolo_width * width
            box_height = yolo_height * height
            x_min = int(x_yolo * width - (box_width / 2))
            y_min = int(y_yolo * height - (box_height / 2))
            x_max = int(x_yolo * width + (box_width / 2))
            y_max = int(y_yolo * height + (box_height / 2))
            if config == DatasetConfig.REAL:
                if class_label_real[cls] == "text":
                    new_img[y_min:y_max, x_min:x_max] = 255
                elif class_label_real[cls] in ["node", "crossing"]:
                    pass
                elif class_label_real[cls] in ["pmos", "nmos"]:
                    upper_region = new_img[y_min : int(y_min + (y_max - y_min) * 0.4), x_min:x_max]
                    lower_region = new_img[int(y_max - (y_max - y_min) * 0.4) : y_max, x_min:x_max]
                    upper_region[:, :, 1] = upper_region[:, :, 0]
                    lower_region[:, :, 1] = lower_region[:, :, 0]
                    upper_region[:, :, 0] = 255
                    lower_region[:, :, 0] = 255
                else:
                    region = new_img[y_min:y_max, x_min:x_max]
                    region[:, :, 2] = region[:, :, 0]
                    region[:, :, 0] = 255
            else:
                region = new_img[y_min:y_max, x_min:x_max]
                region[:, :, 2] = region[:, :, 0]
                region[:, :, 0] = 255
    return new_img


def get_slice(img, data, x, y, w, h, debug):
    new_img = crop_image(img, x, y, w, h)
    scaled_x = x / img.shape[1]
    scaled_y = 1 - y / img.shape[0]
    scaled_w = w / img.shape[1]
    scaled_h = h / img.shape[0]
    scaled_box = box(scaled_x, scaled_y, scaled_x + scaled_w, scaled_y - scaled_h)
    scaled_box_bounds = scaled_box.bounds
    intersection = []
    for line in data:
        line = LineString(line)
        if line.intersects(scaled_box):
            x1, y1, x2, y2 = line.bounds
            x1 = (x1 - scaled_box_bounds[0]) / scaled_w
            y1 = (y1 - scaled_box_bounds[1]) / scaled_h
            x2 = (x2 - scaled_box_bounds[0]) / scaled_w
            y2 = (y2 - scaled_box_bounds[1]) / scaled_h
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))
            x1 = round(x1, 3)
            y1 = round(y1, 3)
            x2 = round(x2, 3)
            y2 = round(y2, 3)
            if norm1((x1 * w, y1 * h), (x2 * w, y2 * h)) < 3:
                continue
            intersection.append([(x1, y1), (x2, y2)])
    if debug:
        plot_images(draw_line(new_img, intersection), img_width=200)
    return new_img, intersection


def get_random_slice(img, data, w, h, debug):
    x = np.random.randint(0, img.shape[1] - w)
    y = np.random.randint(0, img.shape[0] - h)
    return get_slice(img, data, x, y, w, h, debug)


def load_data(img_name, dir, config):
    dir = Path(dir)
    image_dir = dir / Path("images")
    pkl_dir = dir / Path("pkl")
    txt_dir = dir / Path("labels")
    img = cv2.imread(str(image_dir / img_name))
    processed_img = get_box(img, str(txt_dir / img_name.replace(".jpg", ".txt")), config)
    data = pickle.load(open(pkl_dir / img_name.replace(".jpg", ".pkl"), "rb"))
    data = list(chain.from_iterable(data.values()))
    data = np.array(data).reshape(-1, 2, 2)
    return img, processed_img, data


def load_test_data(img, label_name, dir, config):
    dir = Path(dir)
    txt_dir = dir / Path("labels")
    processed_img = get_box(img, str(txt_dir / label_name), config)
    return img, processed_img


if __name__ == "__main__":
    # img_name = list(Path("real_data/train/images").iterdir())[0].name
    # img, processed_img, data = load_data(img_name, "real_data/train")
    img, processed_img, data = load_data("000223.jpg", "real_data/train", DatasetConfig.REAL)
    plot_images(img, img_width=800)
    plot_images(processed_img, img_width=800)
    # for i in range(5):
    #     cropped_img, line_segments = get_random_slice(img, data, 100, 100, debug=True)

# %%
