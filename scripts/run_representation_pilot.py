"""Local/worker CLI; deliberately has no cloud submission path."""
import argparse
from hypercast4d.representation_pilot import run_pair

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--backbone',choices=['micn','film'],required=True)
    p.add_argument('--seed',type=int,choices=[1101,1102,1103],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--epochs',type=int,default=100)
    a=p.parse_args()
    run_pair(a.bundle,a.output,a.backbone,a.seed,a.device,a.epochs)
