import json
import os
from typing import Dict

BASE_FILE_PATH = "test/test_case_backprop/"


class ReaderBackprop:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_backprop(filename: str) -> Dict:
        """ 
        Read file test case
        """
        try:
            print(BASE_FILE_PATH + filename)
            with open(BASE_FILE_PATH + filename) as f:
                data = json.load(f)
                model = data["case"]
                expected = data["expect"]
                return model, expected
        except OSError as e:
            print("File not found")
            os._exit(-1)
