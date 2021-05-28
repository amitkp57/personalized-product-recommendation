import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = 'C:/Users/public.DESKTOP-5H03UEQ/Downloads/venus'


def load_from_csv(csv_name, size=10000):
    """
    Loads data from csv file and save it as .npy file
    :param csv_name:
    :param size:
    :return:
    """
    csv_path = f'{DATA_DIR}/{csv_name}'
    data = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row[:3])
    data = np.array(data)[:size]
    path = Path(csv_path)
    npy_file = f'{DATA_DIR}/{path.stem}.npy'
    with open(npy_file, 'wb') as f:
        np.savez(f, data)
    return


def load_data(name):
    """
    Loads .npy file and returns a numpy array
    :param name:
    :return:
    """
    return np.load(f'{DATA_DIR}/{name}.npy')['arr_0']


def create_dataset(name):
    """
    Loads .npy file and returns a pandas dataframe
    :param name:
    :return:
    """
    ratings = load_data(name)
    ratings_dict = defaultdict(lambda: defaultdict(int))
    for row in ratings:
        user = row[0]
        product = row[1]
        rating = row[2]
        ratings_dict[product][user] = rating
    df = pd.DataFrame.from_dict(ratings_dict, orient='index')
    return df


def save_patio_lawn_garden():
    """
    Read JSON file and saves data in .npy format
    :return:
    """
    review_data = []
    with open(f'{DATA_DIR}/Patio_Lawn_and_Garden_5.json', 'r') as f:
        for row in f:
            review_json = json.loads(row)
            review_data.append([review_json['reviewerID'], review_json['asin'], int(review_json['overall'])])
    npy_file = f'{DATA_DIR}/Patio_Lawn_and_Garden_5.npy'
    with open(npy_file, 'wb') as f:
        np.savez(f, review_data)
    return


if __name__ == '__main__':
    # save_npy('ratings_Apps_for_Android.csv', size=10000)
    # # ratings = load_data('ratings_Apps_for_Android_small')
    # df = create_dataset('ratings_Apps_for_Android').to_numpy(dtype=float)
    # print(df)
    save_patio_lawn_garden()
