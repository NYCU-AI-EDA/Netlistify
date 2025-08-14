# https://ncps.readthedocs.io/en/latest/examples/atari_bc.html
from scipy.ndimage import rotate
from shapely import LineString, box

import main_config as config
from Model import *
from slice import *
from visualizer import get_local


def image_preprocess(img):
    if max(img.shape) > 1000:
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    pad = 20
    half_pad = pad // 2
    img = resize_with_padding(img, pad, pad, 255)
    img = shift(img, (half_pad, half_pad), fill=255)
    return img, pad, half_pad


class FormalDatasetWindowedLinePair(Datasetbehaviour):
    def __init__(self, size, dataset_source, pick, full, direction):
        cache_dir = Path("cache")
        cache_path = cache_dir / Path(dataset_source) / str(size)
        self.cache_path = cache_path
        shutil.rmtree("tmp", ignore_errors=True)
        Path("tmp").mkdir(parents=True, exist_ok=True)
        train_source = Path("train_source")
        shutil.rmtree(str(train_source), ignore_errors=True)
        train_source.mkdir(parents=True, exist_ok=True)
        # shutil.rmtree(cache_path, ignore_errors=True)
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
            self.img_folder = Path(dataset_source) / Path("images")
            self.img_list = list(self.img_folder.iterdir())
            self.img_list = path_like_sort(self.img_list)
            if size > 0:
                self.img_list = self.img_list[:size]
            i = 0
            for img_path in tqdm(self.img_list):
                _, img, data = load_data(
                    img_path.name,
                    dataset_source,
                    config.DatasetConfig.REAL if config.REAL_DATA else config.DatasetConfig.CC,
                )
                data = np.array(data)
                for d in data:
                    angle = self.calculate_line_angle(*d[0], *d[1])
                    if abs(angle - 45) < min(angle, 90 - angle):
                        pass
                    else:
                        if abs(d[0, 0] - d[1, 0]) > abs(d[0, 1] - d[1, 1]):
                            d[0, 1] = d[1, 1] = (d[0, 1] + d[1, 1]) / 2
                        else:
                            d[0, 0] = d[1, 0] = (d[0, 0] + d[1, 0]) / 2
                img, pad, half_pad = image_preprocess(img)
                data[:, :, 0] *= (img.shape[1] - pad) / img.shape[1]
                data[:, :, 1] *= (img.shape[0] - pad) / img.shape[0]
                data[:, :, 0] += half_pad / img.shape[1]
                data[:, :, 1] += half_pad / img.shape[0]
                cv2.imwrite(
                    str(train_source / img_path.name),
                    draw_line(img, data, endpoint=True, endpoint_thickness=5),
                )
                buffer = []
                # slice_image_into_windows(img, 50, -40, buffer=buffer)
                slice_image_into_windows(img, 50, 0, buffer=buffer)
                a_data = [get_slice(img, data, *i, 50, 50, False) for i in buffer]
                a_data = list(
                    filter(
                        lambda x: x[0].shape[0] == 50
                        and x[0].shape[1] == 50
                        and x[0].sum() > 0
                        and x[0].mean() < 252,
                        a_data,
                    )
                )
                # c = [draw_line(i[0], i[1]) for i in a_data]
                # cv2.imwrite(f"tmp/{i}.png", create_grid(c, nrow=12))
                # plot_images(create_grid([i[0] for i in a_data], nrow=12), img_width=400)
                for img, data in a_data:
                    with open(cache_path / str(i), "wb") as f:
                        pickle.dump((img, data), f)
                        i += 1
                # if config.ZOOM:
                #     with open(cache_path / str(i), "wb") as f:
                #         pickle.dump((cv2.resize(img, (0, 0), fx=0.5, fy=0.5), data), f)
                #         i += 1
        size = len(list(cache_path.glob("*")))
        self.pick = pick
        self.direction = direction
        super().__init__(
            size * pick * self.direction,
            self.__create,
            always_reset=True,
        )
        self.data_list = []
        self.full = full
        self.training_data = []

    def calculate_line_angle(self, x1, y1, x2, y2):
        """
        Calculate the angle of a line given two points (x1, y1) and (x2, y2).

        Args:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.

        Returns:
        angle_radians: Angle of the line in radians.
        angle_degrees: Angle of the line in degrees.
        """
        # Calculate the differences
        dx = x2 - x1
        dy = y2 - y1

        # Calculate the angle in radians
        angle_radians = math.atan2(abs(dy), abs(dx))

        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    # def rotate_point(self, point):
    #     rotation_matrix = np.array([[0, -1], [1, 0]])
    #     result = rotation_matrix @ point
    #     result[0] += 1
    #     return result

    def rotate_2d_point(self, point, angle_degrees):
        point = np.asarray(point, dtype=np.float32)
        if angle_degrees == 90:
            rotation_matrix = np.array([[0, -1], [1, 0]])
            rotated_point = rotation_matrix @ point
            rotated_point[0] += 1
        else:
            point -= 0.5
            # print(point)
            # exit()
            theta = np.radians(angle_degrees)
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            rotated_point = rotation_matrix @ point
            # rotated_point[0] += -np.dot(rotation_matrix, np.array([0, 1]))[0]
            rotated_point += 0.5
            # rotated_point += 0.5

        return rotated_point

    def rotate_line(self, line, angle_degrees):
        rline = [self.rotate_2d_point(point, angle_degrees) for point in line]
        return np.array(rline)
        r = 0.1
        rline = LineString(rline).intersection(box(0, 0, 1, 1))
        assert isinstance(rline, LineString)
        if rline.within(box(0, 0, 1, 1) - box(r, r, 1 - r, 1 - r)):
            return np.array([])
        else:
            return np.array(rline.coords)

    def rotate_lines(self, lines, angle_degrees):
        r = [self.rotate_line(line, angle_degrees) for line in lines]
        return [i for i in r if len(i) == 2]

    def mirror_point_x(self, point):
        return np.array([point[0], 1 - point[1]])

    def mirror_point_y(self, point):
        return np.array([1 - point[0], point[1]])

    def mirror_line_x(self, line):
        return [self.mirror_point_x(point) for point in line]

    def mirror_line_y(self, line):
        return [self.mirror_point_y(point) for point in line]

    def mirror_lines_x(self, lines):
        return [self.mirror_line_x(line) for line in lines]

    def mirror_lines_y(self, lines):
        return [self.mirror_line_y(line) for line in lines]

    def rotate_img(self, img, angle_degrees):
        return rotate(img, angle_degrees, cval=255, reshape=False, order=3)

    def augment(self, img, data, v):
        if self.direction == 24:
            if v % 24 == 0:
                img, data = np.flip(img, axis=0), self.mirror_lines_x(data)
            elif v % 24 == 1:
                img, data = np.flip(img, axis=1), self.mirror_lines_y(data)
            elif v % 24 == 2:  # 45
                random_angle = np.random.randint(10, 80)
                img, data = self.rotate_img(img, random_angle), self.rotate_lines(
                    data, random_angle
                )
            elif v % 24 == 3:
                img, data = self.augment(img, data, 2)
                img, data = self.augment(img, data, 0)
            elif v % 24 == 4:
                img, data = self.augment(img, data, 2)
                img, data = self.augment(img, data, 1)
            elif v % 24 == 5:  # 90
                img, data = np.rot90(img), self.rotate_lines(data, 90)
            elif v % 24 == 6:
                img, data = self.augment(img, data, 5)
                img, data = self.augment(img, data, 0)
            elif v % 24 == 7:
                img, data = self.augment(img, data, 5)
                img, data = self.augment(img, data, 1)
            elif v % 24 == 8:  # 135
                img, data = self.augment(img, data, 5)
                img, data = self.augment(img, data, 2)
            elif v % 24 == 9:
                img, data = self.augment(img, data, 8)
                img, data = self.augment(img, data, 0)
            elif v % 24 == 10:
                img, data = self.augment(img, data, 8)
                img, data = self.augment(img, data, 1)
            elif v % 24 == 11:  # 180
                img, data = self.augment(img, data, 5)
                img, data = self.augment(img, data, 5)
            elif v % 24 == 12:
                img, data = self.augment(img, data, 11)
                img, data = self.augment(img, data, 0)
            elif v % 24 == 13:
                img, data = self.augment(img, data, 11)
                img, data = self.augment(img, data, 1)
            elif v % 24 == 14:  # 225
                img, data = self.augment(img, data, 11)
                img, data = self.augment(img, data, 2)
            elif v % 24 == 15:
                img, data = self.augment(img, data, 14)
                img, data = self.augment(img, data, 0)
            elif v % 24 == 16:
                img, data = self.augment(img, data, 14)
                img, data = self.augment(img, data, 1)
            elif v % 24 == 17:  # 270
                img, data = self.augment(img, data, 5)
                img, data = self.augment(img, data, 5)
                img, data = self.augment(img, data, 5)
            elif v % 24 == 18:
                img, data = self.augment(img, data, 17)
                img, data = self.augment(img, data, 0)
            elif v % 24 == 19:
                img, data = self.augment(img, data, 17)
                img, data = self.augment(img, data, 1)
            elif v % 24 == 20:  # 315
                img, data = self.augment(img, data, 17)
                img, data = self.augment(img, data, 2)
            elif v % 24 == 21:
                img, data = self.augment(img, data, 20)
                img, data = self.augment(img, data, 0)
            elif v % 24 == 22:
                img, data = self.augment(img, data, 20)
                img, data = self.augment(img, data, 1)
            elif v % 24 == 23:  # 360
                pass
        elif self.direction == 12:
            if v % 12 == 0:
                img, data = np.rot90(img), self.rotate_lines(data, 90)
            elif v % 12 == 3:
                img, data = self.augment(img, data, 2)
                img, data = self.augment(img, data, 0)
            elif v % 12 == 4:
                img, data = self.augment(img, data, 2)
                img, data = self.augment(img, data, 1)
            elif v % 12 == 5:  # 180
                img, data = np.rot90(img), self.rotate_lines(data, 90)
                img, data = np.rot90(img), self.rotate_lines(data, 90)
            elif v % 12 == 6:
                img, data = self.augment(img, data, 5)
                img, data = self.augment(img, data, 0)
            elif v % 12 == 7:
                img, data = self.augment(img, data, 5)
                img, data = self.augment(img, data, 1)
            elif v % 12 == 8:  # 270
                img, data = np.rot90(img), self.rotate_lines(data, 90)
                img, data = np.rot90(img), self.rotate_lines(data, 90)
                img, data = np.rot90(img), self.rotate_lines(data, 90)
            elif v % 12 == 9:
                img, data = self.augment(img, data, 8)
                img, data = self.augment(img, data, 0)
            elif v % 12 == 10:
                img, data = self.augment(img, data, 8)
                img, data = self.augment(img, data, 1)
            elif v % 12 == 11:  # 360
                pass
        elif self.direction == 4:
            if v % 4 == 0:
                img, data = np.rot90(img), self.rotate_lines(data, 90)
            elif v % 4 == 1:
                img, data = self.augment(img, data, 0)
                img, data = self.augment(img, data, 0)
            elif v % 4 == 2:
                img, data = self.augment(img, data, 0)
                img, data = self.augment(img, data, 0)
                img, data = self.augment(img, data, 0)
            elif v % 4 == 3:
                pass
        elif self.direction == 1:
            pass
        return img, data

    def __create(self, i):
        current = i // self.pick // self.direction
        if current > len(self.data_list) - 1:
            with open(self.cache_path / str(current), "rb") as f:
                self.img, self.data = pickle.load(f)
        if self.full:
            img = cv2.resize(img, (224, 224))
            img, line_segments = self.augment(img.copy(), data.copy(), i)
            # if i < 10:
            #     plot_images(draw_line(img, line_segments), img_width=400)
            # else:
            #     exit()
            return img, line_segments, None
        else:
            try:
                cropped_img, line_segments = self.augment(self.img.copy(), self.data.copy(), i)
                self.training_data.append(draw_line(cropped_img, line_segments).copy())
                if len(self.training_data) == 144:
                    cv2.imwrite(f"tmp/{i}.png", create_grid(self.training_data, nrow=12, padding=5))
                    self.training_data = []
            except ValueError as e:
                print(e)
                return None

            return cropped_img, line_segments


result_num = config.RESULT_NUM


def xtransform(x):
    # gray_channel = transforms.Compose(
    #     [
    #         transforms.ToImage(),
    #         transforms.Grayscale(),
    #         transforms.ToDtype(torch.float32, scale=True),
    #     ]
    # )(x[:, :, :3])
    # alpha_channel = x[:, :, -1] / 255
    # alpha_channel = alpha_channel[np.newaxis, :]
    # joint = torch.cat((gray_channel, torch.tensor(alpha_channel)), dim=0)
    # if config.MODEL_STYLE != "cnn":
    #     joint = transforms.Compose(
    #         [
    #             transforms.ToImage(),
    #             transforms.ToDtype(torch.float32, scale=True),
    #         ]
    #     )(x.copy())
    # else:
    x = png_to_jpg(x)
    joint = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(x)
    # img = np.array(transforms.ToPILImage()(joint))
    # plot_images(img, img_width=400)
    # exit()
    return joint


def ytransform(x):
    s = np.full((result_num, 2, 2), 0, dtype=np.float32)
    if len(x) > 0:
        s[: len(x)] = x
    return torch.tensor(s).float()


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    @get_local("attention_weights")
    def forward(self, src, *args, **kwargs):
        # This is a simplified mimic of _transformer_encoder_layer_fwd
        norm1 = self.norm1
        norm2 = self.norm2
        dropout1 = self.dropout1
        dropout2 = self.dropout2

        # Self-attention block
        src2, attention_weights = self.self_attn(
            src, src, src, attn_mask=kwargs.get("attn_mask"), need_weights=True
        )
        src = src + dropout1(src2)
        src = norm1(src)

        # Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + dropout2(src2)
        src = norm2(src)

        # Return attention weights along with the final output
        return src


class ViT_ex(nn.Module):
    def __init__(
        self,
        patch_size,
        dim,
        depth,
        heads,
        dim_ff,
        result_num,
        dropout=0,
        channels=3,
        style="decoder",
        output_dim=4,
    ):
        super().__init__()
        self.style = style

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # image_height, image_width = image_size, image_size

        # self.transformer = Transformer(dim, depth, heads, dim_head, dim_head, dropout=dropout)
        self.use_mamba = "mamba" in style
        self.use_encoder = "encoder" in style
        self.use_decoder = "decoder" in style
        self.use_moco = "moco" in style

        patch_height, patch_width = patch_size, patch_size

        self.pos_embedding = PositionalEncoding(dim)
        self.transformer = nn.TransformerEncoder(
            CustomTransformerEncoderLayer(dim, heads, dim_ff, batch_first=True, dropout=dropout),
            depth,
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, dim_ff, batch_first=True, dropout=dropout),
            depth,
        )
        self.decoder_query = nn.Embedding(result_num, dim)

        self.box_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, output_dim),
        )
        if config.MODEL_STYLE == "encoder, decoder":
            patch_dim = channels * patch_height * patch_width
            self.to_patch_embedding = nn.Sequential(
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width
                ),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )
        elif config.MODEL_STYLE == "cnn":
            self.cnn = torchvision.models.resnet50(weights="DEFAULT")
            self.cnn.layer4 = nn.Conv2d(1024, 256, 1, 1)
            self.cnn.avgpool = nn.Identity()
            self.cnn.fc = nn.Identity()
            self.cnn_postprocess = nn.Sequential(
                Rearrange("a (b c d) -> a (c d) b", b=256, c=14),
                nn.Linear(256, dim),
            )

    def forward(self, x, y):
        if config.MODEL_STYLE == "encoder, decoder":
            x = x.to(torch.float32)
            x = self.to_patch_embedding(x)
            x += self.pos_embedding(x)
            x = self.transformer(x)
            tgt = repeat(self.decoder_query.weight, "d e -> n d e", n=x.shape[0])
            x = self.decoder(tgt, x)
            x = self.box_head(x)
            x = x.reshape(x.shape[0], -1, 2, 2)
            return x
        elif config.MODEL_STYLE == "cnn":
            x = x.to(torch.float32)
            x = transforms.Resize((224, 224))(x)
            x = self.cnn(x)
            x = self.cnn_postprocess(x)
            x += self.pos_embedding(x)
            x = self.transformer(x)  # Shape: torch.Size([144, 196, 256])
            tgt = repeat(self.decoder_query.weight, "d e -> n d e", n=x.shape[0])
            x = self.decoder(tgt, x)
            x = self.box_head(x)
            return x


# torch.autograd.set_detect_anomaly(True)


def create_model(**kwargs):
    style = config.MODEL_STYLE
    depth = config.DEPTH
    head_num = config.NUM_HEADS
    dim_head = config.EMBED_DIM
    result_num = config.RESULT_NUM
    parameters = {
        "patch_size": config.PATCH_SIZE,
        "dim": head_num * dim_head,
        "depth": depth,
        "heads": head_num,
        "dim_ff": head_num * dim_head,
        "result_num": result_num,
        "channels": 3,
        "dropout": config.DROPOUT,
        "style": style,
        "output_dim": 4 + config.CLASS_OUTPUT,
    }
    parameters |= kwargs
    network = ViT_ex(**parameters)
    return network


def Hungarian_Order(g1b, g2b):
    x1, y1 = g1b[:, :, [0, 1]], g1b[:, :, [2, 3]]
    x2, y2 = g2b[:, :, 0], g2b[:, :, 1]
    C1 = torch.cdist(x1, x2) + torch.cdist(y1, y2)
    C2 = torch.cdist(x1, y2) + torch.cdist(y1, x2)
    C3 = torch.min(C1, C2).cpu().detach()

    indices = [linear_sum_assignment(c)[1] for c in C3]
    for i in range(len(indices)):
        ind = indices[i]
        g2b[i] = g2b[i][ind]
    # 32 15 2 2
    x1, y1 = g1b[:, :, [0, 1]], g1b[:, :, [2, 3]]
    x2, y2 = g2b[:, :, 0], g2b[:, :, 1]
    C1 = torch.abs(x1 - x2) + torch.abs(y1 - y2)
    C1 = C1.sum(dim=2)
    C2 = torch.abs(x1 - y2) + torch.abs(y1 - x2)
    C2 = C2.sum(dim=2)
    min_index = C1 > C2
    g2b[min_index] = g2b[min_index][:, [1, 0]]


def criterion(y_hat, y):
    Hungarian_Order(y_hat, y)
    # print(model.meta.data[0])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
    #     model.meta.data[0].output
    # )
    # plot_images(draw_line(img, y_hat[0]))
    # exit()
    loss_box = F.smooth_l1_loss(y_hat[:, :, [[0, 1], [2, 3]]], y)
    if config.CLASS_OUTPUT:
        class_label = 1 - (y < 1e-8).all(dim=2).all(dim=2).float()
        loss_class = F.binary_cross_entropy_with_logits(y_hat[:, :, 4], class_label)
        return loss_box + 0.01 * loss_class
    else:
        return loss_box

    # for i in range(10):
    #     img = np.array(transforms.ToPILImage()(model.meta.data[i].output))
    #     # plot_images(img, img_width=400)
    #     plot_images(draw_line(img, y_hat[i]), 400)
    # exit()
    # return loss_box


from torchmetrics.classification import BinaryAccuracy

accuracy_metric = BinaryAccuracy().to("cuda")


def eval_metrics(criterion, y_hat, y):
    accs = {}
    loss = criterion(y_hat, y)
    class_label = ~(y < 1e-8).all(dim=2).all(dim=2)
    if config.CLASS_OUTPUT:
        logits = y_hat[:, :, 4]
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).int()
        classification_accuracy = accuracy_metric(predictions, class_label)
        accs["c_acc"] = classification_accuracy.item()

    C = torch.cdist(y_hat[:, :, None, [0, 1]], y[:, :, None, 0]) + torch.cdist(
        y_hat[:, :, None, [2, 3]], y[:, :, None, 1]
    )
    C = C.squeeze()
    C = C[class_label]
    accs["d_acc"] = 0
    # accs["md_acc"] = 0
    if C.numel() > 0:
        distance_accuracy = (C < 0.1).float().mean()
        accs["d_acc"] = distance_accuracy.item()
        # accs["md_acc"] = -C.mean().item()

    return loss, accs


model = None


def main():
    network = create_model()
    set_seed(42, deterministic=False)
    dataset_guise = FormalDatasetWindowedLinePair(
        config.DATASET_SIZE,
        config.DATASET_PATH,
        config.PICK,
        not config.SMALL_IMAGE,
        direction=config.DIRECTION,
    )

    dataset_eval = None
    if config.EVAL_DATASET_PATH is not None:
        dataset_eval = FormalDatasetWindowedLinePair(
            config.EVAL_DATASET_SIZE,
            config.EVAL_DATASET_PATH,
            config.PICK,
            not config.SMALL_IMAGE,
            direction=config.DIRECTION,
        )
    global model
    model = Model(
        dataset_guise,
        dataset_eval,
        xtransform=xtransform,
        ytransform=ytransform,
        amp=False,
        # cudnn_benchmark=True,
        batch_size=config.BATCH_SIZE,
        eval=config.EVAL,
        shuffle=True,
    )

    model.fit(
        network,
        criterion,
        optim.Adam(network.parameters(), lr=config.LEARNING_RATE),
        config.EPOCHS,
        max_epochs=config.MAX_EPOCHS if hasattr(config, "MAX_EPOCHS") else float("inf"),
        pretrained_path=config.PRETRAINED_PATH,
        keep=not config.EVAL,
        backprop_freq=config.BATCH_STEP,
        device_ids=config.DEVICE_IDS,
        eval_metrics=eval_metrics,
        keep_epoch=config.KEEP_EPOCH,
        keep_optimizer=config.KEEP_OPTIMIZER,
        config=(get_attr(config) if not config.EVAL else None),
        upload=config.UPLOAD,
        flush_cache_after_step=config.FLUSH_CACHE_AFTER_STEP,
    )


if __name__ == "__main__":
    main()
