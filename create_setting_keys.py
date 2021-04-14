import json
from argparse import ArgumentParser

from util import create_config_key

def create_keys_from_configs(input_path: str):
    """
    :param input_path: Path to settings.json (must be placed in the same directory as the "datasets"-folder)
    :return:
    """
    # Read settings file
    with open(f'{input_path}') as file:
        settings = json.load(file)

    for setting_key, setting_data in settings.items():
        print(create_config_key(setting_data))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="path to project", metavar="path")
    args = parser.parse_args()
    input_path = args.input
    create_keys_from_configs(input_path)