import os

class BaseCassavaConfig:
    COMPETITION_NAME = 'cassava-leaf-disease-classification'

    BASE_DIR = '.'

    def __init__(self, project_name):

        self.BASE_DIR = self.BASE_DIR + f'/{self.COMPETITION_NAME}'

        self.PROJECT_NAME = project_name
        self.PROJECT_DIR = self.BASE_DIR + '/projects/' + project_name
        self.RESULT_BASE_DIR = self.PROJECT_DIR + '/result'

        # self.__rotate_version()

    def __rotate_version(self):
        
        if os.path.exists(self.RESULT_BASE_DIR):
            # result/配下のversionを更新する。
            result_versions = os.listdir(self.RESULT_BASE_DIR)
            latest_version = sorted(result_versions)[-1]
            prefix, latest_version_number = latest_version.split('_')

            new_version = f'/{prefix}_{int(latest_version_number)+1}'
            os.mkdir(self.RESULT_BASE_DIR + new_version)
        else:
            # result/version_1を作成する。
            os.mkdir(self.RESULT_BASE_DIR + f'/version_1')
    
    def save_config(self):
        pass


class CassavaColabConfig(BaseCassavaConfig):

    BASE_DIR = f'/content/drive/My Drive/kaggle'

    def __init__(self, project_name):
        super().__init__(project_name)

        print(self.BASE_DIR)
        print(self.PROJECT_DIR)

if __name__ == '__main__':
    config = CassavaColabConfig('test')