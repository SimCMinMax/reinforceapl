# -*- coding: utf-8 -*-
"""
Script to run when running reinforceapl in command line.

@author: skasch
"""

import argparse

from .simcenv import SimcEnv


def arparser_args(parser):
    """
    Add arguments to argparse parser
    """
    pass


def main():
    """
    Function to process if ARParser is used as a script.
    """
    parser = argparse.ArgumentParser()
    arparser_args(parser)
    args = parser.parse_args()


main()
