"""
Created by: Gary Kirk

Weather Test
Write various functions processing
CSV files.
"""

import csv
import re
from statistics import mean, stdev
from collections import defaultdict
from typing import Tuple, Dict, List
from itertools import zip_longest
from operator import sub
import numpy as np


def get_avg_n_std_between_suns(mycsv: str, date: str) -> Tuple[any, any]:
    """
    Takes a date as its argument and returns a data structure with the average
    and standard deviation of the temperature (dry-bulb temperature) between
    the hours of sunrise and sunset. In csv Fahr uses integers, Celsius uses float.
    :param mycsv: The csv file.
    :param date:  The date to process.
    :return: Tuple containing the average and standard dev
    """
    with open(mycsv, newline='') as csvfile:
        temperatures = []
        mean_temp, std_dept = None, None
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            if not row['REPORTTPYE'] == 'SOD' and row['DATE'].startswith(date):
                time = row['DATE'].split(' ', 1)[1]  # get time
                time = int(re.sub(':', '', time))  # remove ':', convert to int
                if int(row['DAILYSunrise']) <= time <= int(row['DAILYSunset']):
                    temperatures.append(row['HOURLYDRYBULBTEMPF'])
        # remove letters from numbers 64s, 65r, etc
        temperatures = list(map(lambda z: z[:-1] if not z.isdigit() else z, temperatures))
        temperatures = list(map(int, temperatures))
        if temperatures:
            mean_temp = mean(temperatures)
            std_dept = stdev(temperatures)
        return mean_temp, std_dept


def get_wind_chill_minus_40(mycsv: str, date: str) -> Dict[str, int]:
    """
    Write a method that takes a date as its argument and returns the wind chill
    rounded to the nearest integer for the times when the temperature
    is less than 40 degrees Fahrenheit.
    The wind chill formula, according to the National Weather Service:

    Wind Chill = 35.74 + 0.6215T – 35.75(V^0.16) + 0.4275T(V^0.16)

    where T is the air temperature in Fahrenheit, and V is the wind speed in mph
    :param mycsv: The csv file.
    :param date: The date to process.
    :return: A dictionary of {time : windchill}.
    """
    with open(mycsv, newline='') as csvfile:
        cold_data = dict()
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            if not row['REPORTTPYE'] == 'SOD' and row['DATE'].startswith(date):
                time = row['DATE'].split(' ', 1)[1]
                V = int(row['HOURLYWindSpeed'])
                T = int(row['HOURLYDRYBULBTEMPF'])
                wind_chill = 35.74 + (0.61215 * T) - (35.75 * pow(V, 0.16)) + (0.4275 * T * pow(V, 0.16))
                # print(wind_chill)
                wind_chill = round(wind_chill)
                if wind_chill < 40:
                    cold_data[time] = wind_chill
        return cold_data


DayDict = Dict[str, np.ndarray]


def get_most_similar_day(c1: str, c2: str) -> List[DayDict, ]:
    """
    Write a method that reads both data sets and finds the day in which the conditions
    in Canadian, TX, were most similar to Atlanta's Hartsfield-Jackson Airport.
    You may use any column for your similarity metric, but be prepared to justify
    your choice of measurements.

    AA=DAILYMaximumDryBulbTemp
    AB=DAILYMinimumDryBulbTemp
    AC=DAILYAverageDryBulbTemp

    Comparing AA, AB, AC values as they provide a general summary of the day's
    climate in numerical representation, and each csv file has all of these
    particular fields (for simplicity's sake). The key is to sort the data
    on the numerical fields such that similar fields from each file from can
    be compared adjacently.

    Makes runtime a feasible ~1.5 seconds O(n + nlogn) instead of O(n^2) which,
    on massive data sets like this could take hours to run. The numerical fields
    are vectorized then compared by taking taking the Euclidean distance between
    adjacent vectors. In the code I find the norm of the difference between the
    vectors to get this distance. Then the distance is normalized with respect to
    the norm of one the the original vectors. This function can be easily and efficiently
    expanded to increase accuracy as long as new criteria is represented numerically.

    :param c1: csv file 1
    :param c2: csv file 2
    :return: The most similar day between the two csv files
    """
    d1 = defaultdict(lambda: np.zeros(3, dtype=np.int16))  # csv1 columns AA, AB, AC
    d2 = defaultdict(lambda: np.zeros(3, dtype=np.int16))  # csv2 columns AA, AB, AC

    # Process csv 1
    with open(c1, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            if row['DAILYMaximumDryBulbTemp']:
                date = row['DATE'].split(' ', 1)[0]
                d1[date][0] = int(row['DAILYMaximumDryBulbTemp'])
                d1[date][1] = int(row['DAILYMinimumDryBulbTemp'])
                d1[date][2] = int(row['DAILYAverageDryBulbTemp'])

    # Sort d1 dict on numpy array values, smallest (coldest) to largest (warmest)
    d1 = dict(sorted(d1.items(), key=lambda v: v[1].sum()))
    d1list = []
    # Convert to list of d1 entries
    for i, j in d1.items():
        d1list.append({i: j})
    del d1

    # Process csv 2
    with open(c2, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            if row['REPORTTPYE'] == 'SOD':
                try:
                    date = row['DATE'].split(' ', 1)[0]
                    d2[date][0] = int(row['DAILYMaximumDryBulbTemp'])
                    d2[date][1] = int(row['DAILYMinimumDryBulbTemp'])
                    d2[date][2] = int(row['DAILYAverageDryBulbTemp'])
                except ValueError as vE:
                    pass
                    print('Invalid data entry, record discarded. '
                          'Row:', idx, 'Date:', row['DATE'], vE.args, '\n')

    # Sort d2 dict on numpy array values
    d2 = dict(sorted(d2.items(), key=lambda v: v[1].sum()))
    d2list = []
    for i, j in d2.items():
        d2list.append({i: j})
    del d2

    # Create 2d list of even chunk sizes (size is shorter list)
    size_of_chunk = len(d1list)
    d2chunks = list(d2list[i: i + size_of_chunk] for i in range(0, len(d2list), size_of_chunk))

    # Read through lists of dicts in parallel. Process larger (d2list) in chunks
    # the size of smaller list (d1list). fillvalue contains best encountered day.
    # fill_dict is a placeholder variable. Will hold updates
    # from p (a dict from d1list) when closest value encountered.
    fill_dict = {'-': 0}
    days = [{}, {}]
    closest = 999

    for chunk in d2chunks:
        for p, q in zip_longest(d1list, chunk, fillvalue={tuple(fill_dict.keys())[0]: tuple(fill_dict.values())[0]}):

            # difference between adjacent numpy array vectors
            diff = sum(map(sub, p.values(), q.values()))

            # calculate distance (norm/magnitude/Euclidean_length of difference)
            distance = np.linalg.norm(diff)

            # normalizing distance with respect to norm of original array p
            value = distance / np.linalg.norm(tuple(p.values())[0])

            # always 0 <= value <= 1, if closer to 0 -> similar, else closer to 1 -> different
            if value < closest:
                fill_dict = {tuple(p.keys())[0]: tuple(p.values())[0]}
                closest = value
                days[0] = p
                days[1] = q

    return days


if __name__ == '__main__':

    csv_1 = '1089419.csv'   # Canadian, TX, smaller
    csv_2 = '1089441.csv'   # Atlanta-Hartsfield IA, larger

    """
    Dates must be represented in d/m/yy format with
    no leading zeros (as it is written in the csv files).
    """

    user_date_1 = '1/1/17'
    answer_1 = get_avg_n_std_between_suns(csv_2, user_date_1)
    print('A day between suns on {}:'.format(user_date_1))
    print('Average dry-bulb temp F ADBT:', answer_1[0])
    print('Standard deviation of ADBT:', answer_1[1])
    print()

    user_date_2 = '1/3/17'
    answer_2 = get_wind_chill_minus_40(csv_1, user_date_2)
    print('Wind chill < 40F on {}:'.format(user_date_2))
    print(answer_2)
    print()

    answer_3 = get_most_similar_day(csv_1, csv_2)
    print('Most similar day between Canadian, TX & Atlanta-Hartsfield:')
    print(answer_3)
    print()
