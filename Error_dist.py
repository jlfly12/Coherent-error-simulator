from numpy import random
import inspect


def error_dist(runs):
    # Arbitrary error distribution
    # Randomly outputs a set of values between -1 and 1 with varying probability
    x = 2 * random.rand(runs) - 1
    return 0.55 * x + 0.1 * x ** 5 + 0.35 * x ** 35


def func_str(func):
    # Return input function as a string
    s = inspect.getsource(func).splitlines()
    last = s[len(s)-1]
    return last[11:]