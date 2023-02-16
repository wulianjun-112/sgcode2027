import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
            description='MMDet test (and eval) a model')
    parser.add_argument('--num',type=int,default=100,help='test config file path')
    args = parser.parse_args()
    return args

args = parse_args()
for x in range(args.num):
    print(x)