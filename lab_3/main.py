import os
from dotenv import load_dotenv
from checksum import calculate_checksum, serialize_result
from file_handler import read_csv, read_json
from validate import validate


def main():
    load_dotenv()
    data = read_csv(os.getenv('DATA'))
    patterns = read_json(os.getenv('REGULAR_EXPRESSIONS'))
    list_with_errors = validate(data, patterns)
    serialize_result(os.getenv('VARIANT'), calculate_checksum(list_with_errors))


if __name__ == "__main__":
    main()