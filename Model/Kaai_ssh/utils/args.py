import yaml
from config.main_config import data_path

def parse_args():
    file_name = data_path
    with open(file_name, 'r', encoding="UTF-8") as f:
        temp = yaml.load(f, Loader = yaml.FullLoader)
    return temp
