import json
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    """
    Reads CSV file as a Pandas DataFrame

    :param path: Path to the CSV file
    :return: Pandas DataFrame
    """
    try:
        dataframe = pd.read_csv(path, encoding='utf-16', header=0, sep=';', dtype=str)
        return dataframe
    except FileNotFoundError:
        print(f"The file was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}.")


def read_json(path: str) -> dict[str, str]:
    """
    Reads json file

    :param path: Path to the JSON file
    :return: Dict of [str, str]
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return dict(json.load(f))
    except FileNotFoundError:
        print(f"The file was not found.")
    except Exception as e:
        print(f"An error occurred while reading json the file: {str(e)}.")
