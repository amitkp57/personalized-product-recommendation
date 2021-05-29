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


def json_to_npy(dataset):
    """
    Read JSON file and saves data in .npy format
    :return:
    """
    review_data = []
    with open(f'{DATA_DIR}/{dataset}.json', 'r') as f:
        for row in f:
            review_json = json.loads(row)
            review_data.append([review_json['reviewerID'], review_json['asin'], int(review_json['overall'])])
    npy_file = f'{DATA_DIR}/{dataset}.npy'
    with open(npy_file, 'wb') as f:
        np.savez(f, review_data)
    return


def sparsity(dataset):
    data = create_dataset(dataset).to_numpy(dtype=float)
    n, m = data.shape
    return 1 - np.count_nonzero(np.isnan(data)) / (m * n)


if __name__ == '__main__':
    # save_npy('ratings_Apps_for_Android.csv', size=10000)
    # # ratings = load_data('ratings_Apps_for_Android_small')
    # df = create_dataset('ratings_Apps_for_Android').to_numpy(dtype=float)
    # print(df)
    # json_to_npy('Amazon_Instant_Video_5')
    # json_to_npy('Automotive_5')
    # json_to_npy('Digital_Music_5')
    # json_to_npy('Musical_Instruments_5')
    # json_to_npy('Office_Products_5')
    # json_to_npy('Patio_Lawn_and_Garden_5')
    print(sparsity('Amazon_Instant_Video_5'))
    print(sparsity('Automotive_5'))
    print(sparsity('Digital_Music_5'))
    print(sparsity('Musical_Instruments_5'))
    print(sparsity('Office_Products_5'))
    print(sparsity('Patio_Lawn_and_Garden_5'))
