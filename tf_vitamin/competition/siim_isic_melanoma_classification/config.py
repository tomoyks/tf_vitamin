class Config:
    COMPETITION_NAME = 'siim-isic-melanoma-classification'

    BASE_DIR = f'/content/drive/My Drive/kaggle/{COMPETITION_NAME}'

    # change by each project.
    PROJECT_NAME = 'efn_06'

    INPUT_DIR = BASE_DIR + '/input'
    PROJECT_DIR = BASE_DIR + '/projects/' + PROJECT_NAME
    MODEL_OUTPUT = PROJECT_DIR + '/model.h5'

    TRAIN_CSV = INPUT_DIR + '/train.csv'
    TEST_CSV = INPUT_DIR + '/test.csv'

    VALIDATION_CSV = ""

    def __init__(self, project_name, dataset_name='default'):
        self.PROJECT_NAME = project_name
        self.set_gcs_path(dataset_name)

    def set_gcs_path(self, name='default'):
        ret = self.get_gcs_path(name)
        self.GCS_PATH = ret['GCS_PATH']
        self.TRAIN_FILES = ret['TRAIN_FILES']
        self.TEST_FILES = ret['TEST_FILES']

    @staticmethod
    def get_gcs_path(name='default'):
        # default
        GCS_PATH = 'gs://kds-a3aec024652bcd1f15b0c708f4c002b53dac682b3c37787a5d439986'
        TRAIN_FILES = GCS_PATH + '/tfrecords/train*'
        TEST_FILES = GCS_PATH + '/tfrecords/test*'

        if name == '512x512_with_70k':
            # 512x512-melanoma-tfrecords-70k-images
            GCS_PATH = 'gs://kds-51dc62bf57c4786d5254ad5d5cecdbde9c81887f9d45aeaf4770c7b5'
            TRAIN_FILES = GCS_PATH + '/train*.tfrec'
            TEST_FILES = GCS_PATH + '/test*.tfrec'

        if name == '256x256':
            # melanoma-256x256
            GCS_PATH = 'gs://kds-e12fcdd8602eaeb3e7343f4b322f78239cfa17e4bf5db65058b851ad'
            TRAIN_FILES = GCS_PATH + '/train*.tfrec'
            TEST_FILES = GCS_PATH + '/test*.tfrec'

        ret = {
            'GCS_PATH': GCS_PATH,
            'TRAIN_FILES': TRAIN_FILES,
            'TEST_FILES': TEST_FILES
        }

        return ret
