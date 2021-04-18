"""
Collection of helpful tools for coding. Might be put in a different package later, if there is more use for them.
@author: Nathanael Jöhrmann and others (see comments)
"""

from functools import partial


# based on katja3 https://stackoverflow.com/a/67148852/12143564
class ClassOrInstanceMethod:
    def __init__(self, function):
        self.function = function

    def __get__(self, instance, cls=None):
        return partial(self.function, cls, instance)
