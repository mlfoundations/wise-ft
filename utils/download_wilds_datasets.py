import os, sys
import argparse
import wilds

def main():

    config_datasets = ['fmow', 'iwildcam']
    data_location = os.getenv('DATA_LOCATION')

    for dataset in config_datasets:
        if dataset not in wilds.supported_datasets:
            raise ValueError(f'{dataset} not recognized.')

    print(f'Downloading the following datasets: {config_datasets}')
    for dataset in config_datasets:
        print(f'=== {dataset} ===')
        wilds.get_dataset(
            dataset=dataset,
            root_dir=os.path.expanduser(data_location),
            download=True)


if __name__=='__main__':
    main()
