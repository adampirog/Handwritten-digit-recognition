# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from custom import preprocessing as pr
import pandas as pd
import numpy as np

mnist = pr.import_mnist()

type(mnist)

X, y = mnist["data"], mnist["target"]

X.shape

type(X)

df = pd.DataFrame(X)

df.head()

df.shape

labels = pd.DataFrame(y)

labels.head()

labels.columns=["label"]

labels["label"]=pd.to_numeric(labels["label"])

labels.head()

labels.info()

df = pd.concat([df, labels], axis=1)

df.head()

df.to_csv("MNIST.csv",index=False)

df2 = pd.read_csv("MNIST.csv")

df.head()
