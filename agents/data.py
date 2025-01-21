import json
import os


class Data:
    """
    Data class for loading data from local files.
    """
    def __init__(self, config):
        self.config = config
        self.users = {}
        self.background = {}
        self.prompt = {}

