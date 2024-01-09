#!python3

import sys
from os import chdir
from argparse import ArgumentParser, ArgumentError 
import pandas as pd

from pathlib import Path


def parse_args(argv):
    parser = ArgumentParser(description="Automatic Fire Scars Mapping", epilog="Bye!")

    #--For train128.py or trainAS.py--
    # Dataset1
    parser.add_argument("-tr1", "--data_train_1_path", type=str, required=True, help="Path to the CSV file for dataset 1 training.")
    parser.add_argument("-ev1", "--data_eval_1_path", type=str, required=True, help="Path to the CSV file for dataset 1 evaluation.")
    # Dataset2
    parser.add_argument("-tr2", "--data_train_2_path", type=str, required=False, help="Path to the CSV file for dataset 2 training.")
    parser.add_argument("-ev2", "--data_eval_2_path", type=str, required=False, help="Path to the CSV file for dataset 2 evaluation.")

    #--For evaluation.py--
    parser.add_argument("-te1", "--data_test_1_path", type=str, required=True, help="Path to the CSV file for dataset 1 testing.")
    parser.add_argument("-te2", "--data_test_2_path", type=str, required=False, help="Path to the CSV file for dataset 2 testing.")
    parser.add_argument("-mp", "--model_path", type=str, required=True, help="Path to the pre-trained model file.(Depends on Model Size value)")
    parser.add_argument("-ms", "--model_size", type=str, choices=["AS", "128"], required=True, help="Model size (AS or 128).")

    args = parser.parse_args(argv)

    # Lógica para manejar valores predeterminados si no se especifican
    if not args.data_train_2_path:
        args.data_train_2_path = pd.DataFrame()
    if not args.data_eval_2_path:
        args.data_eval_2_path = pd.DataFrame()
    if not args.data_test_2_path:
        args.data_test_2_path = args.data_test_1_path

    # Lógica para determinar qué archivo usar según el tamaño del modelo
    if args.model_size == "AS":
        args.train_script = "trainAS.py"
        args.eval_script = "evaluation.py"
    elif args.model_size == "128":
        args.train_script = "train128.py"
        args.eval_script = "evaluation.py"

    return args



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    print(args, Path().cwd())
    chdir(args.change_directory)
    print(args, Path().cwd())

if __name__ == "__main__":
    sys.exit(main())
