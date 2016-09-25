Implementation of (https://arxiv.org/pdf/1511.06856v2.pdf) in tensorflow.

Helps initialize weights for big nets so they can train without things like batch normalization.

Not very well formatted but I thought I'd share it. Included is some sample code on how to use it.

Requires tensorflow, sklearn, re, and numpy

run python magic_init.py to test it out. Look at the code under the "if __name__ == '__main__'" block to see how to use it
