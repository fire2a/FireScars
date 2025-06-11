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
    #data_train_1
    #data_eval_1
    #data_train_2 (optional)
    #data_eval_2 (optional)
    #model_path (optional?)

    return parser.parse_args()
'''
#-----needed arguments-----
only evaluation                 both                        only training
- dataset for evaluation 1 
- dataset for evaluation 2 
- Model size (AS or 128)
                                - Model Path
                                                            - number of epochs
                                                            - batch size
                                                            - learning rate
                                                            - dataset for training 1
                                                            - dataset for training 2 
                                                            - dataset for testing 1
                                                            - dataset for testing 2 
                                - firescars path
                                - img prefire paths
                                - img post fire path
'''


