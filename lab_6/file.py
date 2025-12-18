import json
import argparse


def parse() -> argparse.Namespace:
    """
    Handle argument parsing and return them
    Raises SyntaxError if no filenames is provided
    """
    try:
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "-gen",
            "--generation",
            nargs="?",
            const=128,
            type=int,
            choices=[64, 128, 192],
        )
        group.add_argument("-enc", "--encryption", action="store_true")
        group.add_argument("-dec", "--decryption", action="store_true")
        return parser.parse_args()
    except:
        raise SyntaxError("Path is empty")


def open_file(name) -> str:
    """
    Open and read the input file
    Raises FileNotFoundError if the file cannot be found
    """
    try:
        with open(name, "r", encoding="UTF-8") as file:
            return file.read()
    except:
        raise FileNotFoundError("File doesnt found or doesnt exist")


def save_file(path: str, output: str) -> None:
    """
    Open and write the output file
    Raises Exception if the file cannot be found
    """
    try:
        with open(path, "w", encoding="UTF-8") as file:
            file.write(output)
    except:
        raise Exception("Cannot create output file")


def open_binary_file(name) -> bytes:
    """
    Open and read the binary file
    Raises FileNotFoundError if the file cannot be found
    """
    try:
        with open(name, "rb") as file:
            return file.read()
    except:
        raise FileNotFoundError("Binary file doesnt found or doesnt exist")


def save_binary_file(path: str, output: bytes) -> None:
    """
    Open and write the binary file
    Raises Exception if the file cannot be found
    """
    try:
        with open(path, "wb") as file:
            file.write(output)
    except:
        raise Exception("Cannot create binary file")


def open_json_file(name: str) -> dict:
    """
    Open and read the content of the JSON file
    Raises FileNotFoundError if the JSON file cannot be found
    """
    try:
        with open(name, "r", encoding="UTF-8") as file:
            return json.load(file)
    except:
        raise FileNotFoundError("File doesnt found or doesnt exist")
