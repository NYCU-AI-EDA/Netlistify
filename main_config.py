import platform

from utility import *

machine_name = platform.node()
KEEP_OPTIMIZER = True
KEEP_EPOCH = False
LEARNING_RATE = 1e-4
EPOCHS = 1000
BATCH_SIZE = 64
BATCH_STEP = 1
DEVICE_IDS = [0]
DROPOUT = 0
EVAL_DATASET_PATH = None
FLUSH_CACHE_AFTER_STEP = 0
EVAL = False
STYLE_OPTIONS = ["encoder, decoder", "cnn"]
MODEL_STYLE = STYLE_OPTIONS[1]
REAL_DATA = False #choose dataset, false is the schematic images on https://huggingface.co/datasets/hanky2397/schematic_images.
SMALL_IMAGE = True # Segment an image into small cells 
CLASS_OUTPUT = False # Transformer output with class
UPLOAD = False
WANDB_KEY = "" # Wandb key


class DatasetConfig(Enum):
    CC = auto()
    REAL = auto()


def get_best_model_path(config, class_output=False):
    # choose pt for testing. CC is the dataset on https://huggingface.co/datasets/hanky2397/schematic_images. REAL is teh dataset on your own. 
    global IMAGE_SIZE, RESULT_NUM, CLASS_OUTPUT
    if config == DatasetConfig.CC:
        RESULT_NUM = 35
        IMAGE_SIZE = 50 
        return "runs/FormalDatasetWindowedLinePair/1123_18-10-31/best_train.pth"
    elif config == DatasetConfig.REAL:
        if class_output:
            CLASS_OUTPUT = True
            RESULT_NUM = 10
            IMAGE_SIZE = 50
            return "runs/FormalDatasetWindowedLinePair/1118_17-28-26/best_train.pth"
        else:
            CLASS_OUTPUT = False
            RESULT_NUM = 10
            IMAGE_SIZE = 50
            print(os.getcwd())
            # input()
            # return "runs/1118_20-09-56/best_train.pth"
            return "runs/FormalDatasetWindowedLinePair/0113_14-34-52/best_train.pth"
            
    else:
        raise ValueError("Invalid config")


if MODEL_STYLE == STYLE_OPTIONS[1]:
    if SMALL_IMAGE:
        DATASET_PATH = "/home/111/hank/img2hspice/open_source/cc_deathate_data/train"
        DATASET_SIZE = 300
        PRETRAINED_PATH = ""
        PICK = 1
        FLUSH_CACHE_AFTER_STEP = 0
        IMAGE_SIZE = 50 #cell size
        PATCH_SIZE = 10
        DEPTH = 6
        NUM_HEADS = 8
        EMBED_DIM = 32
        RESULT_NUM = 35
        TEST_DATASET_PATH = "final_test"
        DIRECTION = 1
        if REAL_DATA:
            # PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1117_23-54-55/latest.pth"
            PRETRAINED_PATH = ""
            RESULT_NUM = 10
            PICK = 1
            IMAGE_SIZE = 50
            DATASET_PATH = "real_data/train"
            TEST_DATASET_PATH = "real_data/test"
            EVAL_DATASET_PATH = None
            DATASET_SIZE = -1
            DIRECTION = 1
            DEPTH = 6
            NUM_HEADS = 8
            EMBED_DIM = 32
    else:
        DATASET_SIZE = 30000
        DATASET_PATH = "cc_deathate_data/train"
        EVAL_DATASET_PATH = None
        PRETRAINED_PATH = ""
        PICK = 1
        FLUSH_CACHE_AFTER_STEP = 0
        IMAGE_SIZE = -1
        DEPTH = 6
        NUM_HEADS = 8
        EMBED_DIM = 32
        RESULT_NUM = 125
        TEST_DATASET_PATH = "cc_deathate_data/train"
        DIRECTION = 1
        if REAL_DATA:
            RESULT_NUM = 125
            DATASET_PATH = "real_data/train"
            EVAL_DATASET_PATH = None
            TEST_DATASET_PATH = "real_data/train"
            DATASET_SIZE = -1
