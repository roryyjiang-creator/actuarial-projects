import urllib.request
import pandas as pd

URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
urllib.request.urlretrieve(URL, "data/insurance.csv")
df = pd.read_csv("data/insurance.csv")
print(df.shape)
print(df.dtypes)
print(df.head())
