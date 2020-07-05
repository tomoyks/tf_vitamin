import os
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.rest import ApiException


class KaggleApiWrapper:
    def __init__(self, competition_id, config_dir='/content/drive/My Drive/kaggle'):
        self.competition_id = competition_id
        self.api = self.init_api(config_dir)

    @staticmethod
    def init_api(config_dir):
        os.environ['KAGGLE_CONFIG_DIR'] = config_dir
        api = KaggleApi()
        api.authenticate()

        return api

    def submit_prediction(self, file_path, message):
        try:
            sub_result = self.api.competition_submit(file_path, message, self.competition_id)
            print(sub_result)
        except ApiException:
            return False

        return True
