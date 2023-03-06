from datasets.Dataset import DATASET
from config.main_config import data_path
from utils.args import parse_args
def load_dataset(dataset_name, type = 'Train'):
    """Loads the dataset."""

    implemented_datasets = ('Dataset','HEVI',)
    assert dataset_name in implemented_datasets

    dataset = None
    
    if dataset_name == 'Dataset':
        dataset = DATASET()
    elif dataset_name == 'HEVI':
        args = parse_args()
        dataloader_params ={
        "batch_size": args['batch_size'],
        "shuffle": args['shuffle'],
        "num_workers": args['num_workers']
    }
        dataset = DATASET(args, )