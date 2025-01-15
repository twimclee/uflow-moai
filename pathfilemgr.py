import os
import yaml

class MPathFileManager:
	def __init__(self, volume, project, subproject, task, version):
		self.volume = volume
		self.project = project
		self.subproject = subproject
		self.task = task
		self.version = version

		self.project_path = None
		self.subproject_path = None
		self.task_path = None
		self.version_path = None
		self.weight_path = None
		self.train_result = None
		self.evaluate_result = None
		self.test_dataset = None
		self.test_result = None
		self.train_path = None
		self.val_path = None

		self.hyp_yaml = None
		self.data_yaml = None
		self.result_csv = None

		self.__make_dirs()

	def __make_dirs(self):
		self.project_path = f'/{self.volume}/{self.project}'
		# self.project_path = f'../{self.project}'
		self.__make_dir(self.project_path)

		self.subproject_path = f'{self.project_path}/{self.subproject}'
		self.__make_dir(self.subproject_path)

		self.task_path = f'{self.subproject_path}/{self.task}'
		self.__make_dir(self.task_path)

		self.version_path = f'{self.task_path}/{self.version}'
		self.__make_dir(self.version_path)

		self.weight_path = f'{self.version_path}/weights'
		self.__make_dir(self.weight_path)

		self.train_result = f'{self.version_path}/training_result'
		self.__make_dir(self.train_result)

		self.evaluate_result = f'{self.train_result}/evaluate_result'
		self.__make_dir(self.evaluate_result)    

		self.test_dataset = f'{self.version_path}/inference_dataset'
		self.__make_dir(self.test_dataset)

		self.test_result = f'{self.version_path}/inference_result'
		self.__make_dir(self.test_result)

		self.train_dataset = f'{self.task_path}/train_dataset'
		self.train_path = f'{self.train_dataset}/train'
		self.val_path = f'{self.train_dataset}/valid'

		self.hyp_yaml = f'{self.task_path}/train_dataset/hyp.yaml'
		self.data_yaml = f'{self.task_path}/train_dataset/data.yaml'
		self.result_csv = f'{self.train_path}/results.csv'

		self.test_hyp_yaml = f'{self.train_result}/hyp.yaml'
		self.test_data_yaml = f'{self.train_result}/data.yaml'

	def __make_dir(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

	def load_train_hyp(self, mcls):
		self.__load_yaml(self.hyp_yaml, mcls)

	def load_train_data(self, mcls):
		self.__load_yaml(self.data_yaml, mcls)

	def load_test_hyp(self, mcls):
		self.__load_yaml(self.test_hyp_yaml, mcls)

	def load_test_data(self, mcls):
		self.__load_yaml(self.test_data_yaml, mcls)

	def __load_yaml(self, path, mcls):
		with open(path, 'r') as file:
			config = yaml.safe_load(file)
        
		for key, value in config.items():
			setattr(mcls, key, value)


	def save_hyp(self, mcls):
		self.__save_yaml(f'{self.train_result}/hyp.yaml', mcls)

	def save_data(self, mcls):
		self.__save_yaml(f'{self.train_result}/data.yaml', mcls)

	def __save_yaml(self, path, mcls):
		with open(path, 'w') as f:
			yaml.dump(mcls.__dict__, f, default_flow_style=False, allow_unicode=True)

