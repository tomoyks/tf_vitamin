class Config:
    COMPETITION_NAME = 'siim-isic-melanoma-classification'

    BASE_DIR = f'/content/drive/My Drive/kaggle/{COMPETITION_NAME}'

    INPUT_DIR = BASE_DIR + '/input'

    TRAIN_CSV = INPUT_DIR + '/train.csv'
    TEST_CSV = INPUT_DIR + '/test.csv'

    VALIDATION_CSV = ""

    def __init__(self, project_name, dataset_name='default'):
        self.PROJECT_NAME = project_name
        self.PROJECT_DIR = self.BASE_DIR + '/projects/' + project_name
        self.MODEL_OUTPUT = self.PROJECT_DIR + '/model.h5'

        self.set_gcs_path(dataset_name)

    def set_gcs_path(self, name='default'):
        ret = self.get_gcs_path(name)
        self.GCS_PATH = ret['GCS_PATH']
        self.TRAIN_FILES = ret['TRAIN_FILES']
        self.TEST_FILES = ret['TEST_FILES']

    @staticmethod
    def get_gcs_path(name='default'):
        # default
        GCS_PATH = 'gs://kds-fd4529477b6bfc079d14d2c1fb860b2da18e0293b0c3c51476182df1'
        TRAIN_FILES = GCS_PATH + '/tfrecords/train*'
        TEST_FILES = GCS_PATH + '/tfrecords/test*'

        if name == '512x512_with_70k':
            # 512x512-melanoma-tfrecords-70k-images
            GCS_PATH = 'gs://kds-440c33710adcb2af797fe26a547a54a2f370013a41e3529ac09d8501'
            TRAIN_FILES = GCS_PATH + '/train*.tfrec'
            TEST_FILES = GCS_PATH + '/test*.tfrec'

        if name == '256x256':
            # melanoma-256x256
            GCS_PATH = 'gs://kds-65548a4c87d02212371fce6e9bd762100c34bf9b9ebbd04b0dd4b65b'
            TRAIN_FILES = GCS_PATH + '/train*.tfrec'
            TEST_FILES = GCS_PATH + '/test*.tfrec'

        ret = {
            'GCS_PATH': GCS_PATH,
            'TRAIN_FILES': TRAIN_FILES,
            'TEST_FILES': TEST_FILES
        }

        return ret
