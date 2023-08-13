import tensorflow as tf


def scheduler_1(epoch, lr):
    initial = 0.0001
    if (epoch <= 30):
        return initial  # warmup phase
    else:
        lr = ((0.95)**(epoch-30))*initial
        return lr
