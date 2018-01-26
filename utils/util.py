from __future__ import print_function

import time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def Now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def INFO(*args):

    print(bcolors.OKGREEN, Now(), *args, bcolors.ENDC)


def WARN(*args):
    print(bcolors.WARNING, Now(), *args, bcolors.ENDC)


def FAIL(*args):
    print(bcolors.WARNING, Now(), *args, bcolors.ENDC)
