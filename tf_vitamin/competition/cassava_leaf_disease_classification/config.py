import json
import pathlib
from collections import defaultdict


class BaseCassavaConfig:
    COMPETITION_NAME = 'cassava-leaf-disease-classification'

    BASE_DIR = pathlib.Path('.')

    def __init__(self, project_name):

        self.BASE_DIR = self.BASE_DIR / self.COMPETITION_NAME

        self.PROJECT_NAME = project_name
        self.PROJECT_DIR = self.BASE_DIR / 'projects' / project_name
        self.RESULT_BASE_DIR = self.PROJECT_DIR / 'result'
        self.RESULT_OUTPUT_DIR = None

        self.__rotate_version()
        self.parameter = self.__init_parameter()

    def __rotate_version(self):
        prefix = 'version'

        if not self.RESULT_BASE_DIR.exists():
            self.RESULT_BASE_DIR.mkdir(parrents=True)

        # result/配下のversionを更新する。
        result_versions = list(pathlib.Path(
            self.RESULT_BASE_DIR).iterdir())

        if len(result_versions) == 0:
            new_version = 'version_1'
        else:
            latest_version_number = max(
                [int(x.name.split('_')[1]) for x in result_versions])
            new_version = f'{prefix}_{int(latest_version_number)+1}'

        # result/version_{n}を作成する。
        self.RESULT_OUTPUT_DIR = self.RESULT_BASE_DIR / new_version
        pathlib.Path(self.RESULT_OUTPUT_DIR).mkdir(parents=True)

    def __init_parameter(self):
        parameter = {
            'base_dir': str(self.BASE_DIR),
            'project_dir': str(self.PROJECT_DIR),
            'result_output_dir': str(self.RESULT_OUTPUT_DIR)
        }
        return parameter

    def save_config(self):
        save_path = self.RESULT_OUTPUT_DIR / 'parameter.json'
        with open(save_path, 'w') as json_file:
            json.dump(self.parameter, json_file, indent=2)

    def load_confiig(self, path):
        with open(path, 'r') as json_file:
            self.parameter = json.load(json_file)


class CassavaColabConfig(BaseCassavaConfig):

    BASE_DIR = f'/content/drive/My Drive/kaggle'

    def __init__(self, project_name):
        super().__init__(project_name)


class CassavaKaggleNotebookConfig(BaseCassavaConfig):

    BASE_DIR = f'/kaggle/working'
    INPUT_DIR = f'/kaggle/input'

    def __init__(self, project_name):
        super().__init__(project_name)

        self.PROJECT_DIR = self.BASE_DIR + '/projects/' + project_name
        self.RESULT_BASE_DIR = self.PROJECT_DIR + '/result'


if __name__ == '__main__':
    config = BaseCassavaConfig('test')
    config.save_config()

    config.load_confiig(
        'cassava-leaf-disease-classification/projects/test/result/version_1/parameter.json')

    print(config.parameter)
