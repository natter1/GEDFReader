"""
This file contains helper classes for GDEFImporter to mirror data structures in a raw *.gdf file.
A separate file is necessary, to prevent circular import (e.g. in gdef_measurement.py).
@author: Nathanael Jöhrmann
"""
from enum import Enum
from typing import Optional, List


class GDEFVariableType(Enum):
    VAR_INTEGER = 0
    VAR_FLOAT = 1
    VAR_DOUBLE = 2
    VAR_WORD = 3
    VAR_DWORD = 4
    VAR_CHAR = 5
    VAR_STRING = 6
    VAR_DATABLOCK = 7
    VAR_NVARS = 8  # number of known GDEF types


type_sizes = [4, 4, 8, 2, 4, 1, 0, 0]


class GDEFHeader:
    def __init__(self):
        self.magic = None
        self.version = None
        self.creation_time = None
        self.description_length = None
        self.description = None


class GDEFControlBlock:
    """
    Class attribute _counter is used to generate an ID for each measurement. Has to be reset again, when importing
    a new file, or ids become ambiguous. To prevent unwanted behavior, lock_counter should be set true until imprting
    one file is finished.
    """
    _counter = 0  # total number of created _blocks
    lock_counter = False  # don't allow counter reset

    def __init__(self):
        GDEFControlBlock._counter += 1
        self.id = GDEFControlBlock._counter
        self.mark = None
        self.n_variables = None
        self.n_data = None

        self.variables: List[GDEFVariable] = []
        self.next_byte = None

    @classmethod
    def reset_counter(cls):
        assert not cls.lock_counter, "Can't reset _counter while lock_counter is nit released"
        cls._counter = 0


class GDEFVariable:
    def __init__(self):
        self.name: str = ''
        self.type: Optional[GDEFVariableType] = None
        self.size = None
        self.data = None
