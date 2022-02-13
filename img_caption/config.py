import os
import yaml
import pathlib
# path = os.path.dirname(os.path.realpath(__file__))
# path = pathlib.Path().resolve().parent.absolute()
path = pathlib.Path().resolve()


def get_config(path_to_config):
    """
    Get config
    :param path_to_config: path to config file
    :return: config
    """
    with open(path_to_config, mode="r") as file:
        config = yaml.safe_load(file)

    path_to_data_folder = os.path.join(path, config["data"]["path_to_data_folder"])
    config["data"]["path_to_data_folder"] = path_to_data_folder
    config["data"]["path_to_caption_file"] = os.path.join(path_to_data_folder, config["data"]["caption_file_name"])
    config["data"]["path_to_images"] = os.path.join(path_to_data_folder, config["data"]["images_folder_name"])
    path_to_output_folder = os.path.join(path, config["data"]["output_folder_name"])
    config["data"]["path_to_output_folder"] = path_to_output_folder
    config["data"]["path_to_log_file"] = os.path.join(path_to_output_folder, config["data"]["logging_file_name"])

    return config
