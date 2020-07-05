class ProjectConfig:
    COMPETITION_NAME = 'siim-isic-melanoma-classification'

    BASE_DIR = f'/content/drive/My Drive/kaggle/{COMPETITION_NAME}'

    # change by each project.
    PROJECT_NAME = ''

    INPUT_DIR = BASE_DIR + '/input'
    PROJECT_DIR = BASE_DIR + '/projects/' + PROJECT_NAME
    MODEL_OUTPUT = PROJECT_DIR + '/model.h5'

    TRAIN_CSV = INPUT_DIR + '/train.csv'
    TEST_CSV = INPUT_DIR + '/test.csv'

    GCS_PATH = 'gs://kds-a3aec024652bcd1f15b0c708f4c002b53dac682b3c37787a5d439986'
    TRAIN_FILES = GCS_PATH + '/tfrecords/train*'
    TEST_FILES = GCS_PATH + '/tfrecords/test*'

    VALIDATION_CSV = ""