import torch
from src.train import train
from src.infer import infer
from src.test import test
print("Use CUDA:", torch.cuda.is_available())
print("torch version:", torch.__version__)


# TODO: Implement
def my_train():
    train()

def my_infer():
    infer()

def my_test():
    test()

if __name__ == "__main__":
    # my_train()
    my_infer()
