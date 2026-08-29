"""
Configuration for DINOv2 + Region-Guided Attention knee MRI model.
"""
import os


class Config:
    # -------------------------------------------------------------------------
    # Paths — auto-detect Kaggle vs local
    # -------------------------------------------------------------------------
    IS_KAGGLE = os.path.exists("/kaggle")

    # Kaggle paths
    KAGGLE_VOLUMES_DIR = "/kaggle/input/datasets/barun2104/rsna-knee-mri-processed-3d-volumes"
    KAGGLE_COMP_DIR = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
    KAGGLE_LABELS_PATH = "/kaggle/input/datasets/nishkarshsinghal/train-data-with-labels/train_folds_sota_qwen25.csv"
    KAGGLE_OUTPUT_DIR = "/kaggle/working"

    # Local paths
    LOCAL_VOLUMES_DIR = r"D:\Coding\kaggle\RSNA\data"
    LOCAL_COMP_DIR = r"D:\Coding\kaggle\RSNA\data\main_data"
    LOCAL_LABELS_PATH = r"D:\Coding\kaggle\RSNA\data\main_data\train_folds_sota_qwen25.csv"
    LOCAL_OUTPUT_DIR = r"D:\Coding\kaggle\RSNA\output"

    @classmethod
    def volumes_dir(cls):
        return cls.KAGGLE_VOLUMES_DIR if cls.IS_KAGGLE else cls.LOCAL_VOLUMES_DIR

    @classmethod
    def labels_path(cls):
        return cls.KAGGLE_LABELS_PATH if cls.IS_KAGGLE else cls.LOCAL_LABELS_PATH

    @classmethod
    def series_path(cls):
        d = cls.KAGGLE_COMP_DIR if cls.IS_KAGGLE else cls.LOCAL_COMP_DIR
        return os.path.join(d, "train_series.csv")

    @classmethod
    def output_dir(cls):
        d = cls.KAGGLE_OUTPUT_DIR if cls.IS_KAGGLE else cls.LOCAL_OUTPUT_DIR
        os.makedirs(d, exist_ok=True)
        return d

    # -------------------------------------------------------------------------
    # DINOv2 backbone
    # -------------------------------------------------------------------------
    DINO_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
    DINO_EMBED_DIM = 384
    DINO_CHUNK_SIZE = 32  # slices processed through DINOv2 at once

    # -------------------------------------------------------------------------
    # Targets (12 knee pathologies)
    # -------------------------------------------------------------------------
    TARGET_COLS = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    NUM_TARGETS = 12

    # -------------------------------------------------------------------------
    # Region grouping (anatomical compartments)
    # -------------------------------------------------------------------------
    REGION_GROUPS = {
        "ligament": ["ACL", "MCL"],
        "meniscus": ["Medial Meniscus", "Lateral Meniscus"],
        "oa":       ["Medial OA", "Lateral OA", "PF OA"],
        "fluid":    ["Effusion", "Synovitis", "Baker's"],
        "trauma":   ["Contusion", "Fracture"],
    }

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    MAX_SLICES_PER_SERIES = 16
    MAX_SERIES_PER_STUDY = 3
    MAX_TOTAL_SLICES = MAX_SLICES_PER_SERIES * MAX_SERIES_PER_STUDY  # 48
    IMG_SIZE = 224

    PLANE_MAP = {"Sagittal": 0, "Axial": 1, "Coronal": 2}
    NUM_PLANES = 3
    NUM_SEQ_TYPES = 2  # fluid-sensitive or not

    # -------------------------------------------------------------------------
    # Training hyper-parameters
    # -------------------------------------------------------------------------
    NUM_FOLDS = 5
    TRAIN_FOLD = 0          # fold to use as validation
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 4    # effective batch = 16
    NUM_EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    SEED = 42

    # Shared transformer encoder
    ENCODER_LAYERS = 2
    ENCODER_HEADS = 6
    ENCODER_DROPOUT = 0.1

    # Region attention heads
    REGION_NUM_QUERIES = 4
    REGION_NUM_HEADS = 4
    REGION_DROPOUT = 0.1
