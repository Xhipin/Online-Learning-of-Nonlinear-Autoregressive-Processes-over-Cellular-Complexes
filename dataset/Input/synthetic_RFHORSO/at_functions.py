import numpy as np

def func1():
    t = 0
    while True:
        t+=1
        Phi = yield  # This will receive the new matrix Phi
        if Phi == "reset":  # Check if the input is the reset command
            t = 0
            continue  # Skip the rest of the loop and wait for the next input
        yield 1/ t

functionDatabase = [func1]