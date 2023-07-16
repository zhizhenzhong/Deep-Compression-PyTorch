import argparse

import torch

from net.huffmancoding import huffman_encode_model
from pruning import test

import util

parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
parser.add_argument('--model', type=str, help='saved quantized model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--log', type=str, default='log.txt', help='log file name')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

# model = torch.load(args.model)
model = "saves/model_lenet_after_retraining.ptmodel"
huffman_encode_model(model)
torch.save(model, f"saves/model_lenet_after_hoffman.ptmodel")
accuracy = test()
util.log(args.log, f"accuracy_after_hoffman {accuracy}")