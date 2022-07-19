from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 데이터 불러오기
(train_data,train_label),(test_data,test_label)=keras.datasets.fashion_mnist.load_data()

