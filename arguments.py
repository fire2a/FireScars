from os import chdir
from argparse import ArgumentParser, ArgumentError 
import pandas as pd
from pathlib import Path
import argparse

def get_evaluation_args(argv):
    parser = argparse.ArgumentParser(description='Arguments for evaluation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-ev1', type=str, help='Path to dataset CSV for evaluation 1')
    parser.add_argument('-ev2', type=str, help='Path to dataset CSV for evaluation 2')
    parser.add_argument('-ms', type=str, required=False, help='Model size ("128" or "AS")')
    parser.add_argument('-mp', type=str, help='Path to the trained model')
    return parser.parse_args(argv)

def get_train_args():
    parser = argparse.ArgumentParser(description='Argument parser for train128copia.py or trainAScopia.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-ep', type=int, default=25, help='Number of epochs')
    parser.add_argument('-bs', type=int, default=16, help='Batch size')
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate')
    return parser.parse_args()


