import pandas as pd
import os
import params
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from neuron import Neuron
from connections import Connections

class Test:
    def __init__(self):
        self.attribute = [0,1,2,3,4,5,6,7,8,9]

    def pop(self):
        self.attribute.pop()

class Test2:
    def __init__(self, list):
        self.own_list = list

    def update(self):
        for t_ in self.own_list:
            t_.pop()

    def display(self):
        for t_ in self.own_list:
            print(t_.attribute)
        

t1 = Test()
t2 = Test()
t_list = [t1, t2]
t3 = Test2(t_list)
t3.update()
t3.display()

