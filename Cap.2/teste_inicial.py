import os 
import tarfile 
import urllib.request
import pandas
import matplotlib.pyplot as plt
import numpy as np

DOWLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/refs/heads/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pandas.read_csv(csv_path)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# fetch_housing_data()
housing = load_housing_data()
# print(housing.head())
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")