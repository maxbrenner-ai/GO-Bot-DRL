# Takes any number of csvs and finds avg of each csvs corresponding column row by row

# E.G.
'''
Input (Any number of CSVs with same dim.):

CSV_1:
A1, B1, C1
A2, B2, C2
A3, B3, C3

CSV_2:
A1, B1, C1
A2, B2, C2
A3, B3, C3

CSV_3:
A1, B1, C1
A2, B2, C2
A3, B3, C3

Output:

CSV file of format: N is num of input CSVs (Just stacking and then dividing)
1.A1 + 2.A1 + 3.A1 / N, 1.B1 + 2.B1 + 3.B1 / N, 1.C1 + 2.C1 + 3.C1 / N,
1.A2 + 2.A2 + 3.A2 / N, 1.B2 + 2.B2 + 3.B2 / N, 1.C2 + 2.C2 + 3.C2 / N,
1.A3 + 2.A3 + 3.A3 / N, 1.B3 + 2.B3 + 3.B3 / N, 1.C3 + 2.C3 + 3.C3 / N,
'''

import csv
import numpy as np

directory_path = 'data/case_3_response_to_done/'
num_inputs = 10
input_file_paths = []

for i in range(num_inputs):
    input_file_paths.append(directory_path+'file_'+str(i)+'.csv')

output_file_path = directory_path+'avg.csv'

list_of_tables = []

degree = 0
cardinality = 0
p = 0
for path in input_file_paths:
    list_of_tables.append([])
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            if p == 0:
                degree = len(row)
            list_of_tables[p].append(row)
            if p == 0:
                cardinality += 1
    p += 1

# Temp 'csv' file: list of numpy arrays
output = [np.zeros(shape=(degree,), dtype=np.float32) for _ in range(cardinality)]

# Sum
for table in list_of_tables:
    r = 0
    for row in table:
        output[r] += np.array(row, dtype=np.float32)
        r += 1

# Now AVG
for row in output:
    row /= num_inputs

with open(output_file_path, 'a', newline='') as csvfile:
    wr = csv.writer(csvfile)
    for row in output:
        wr.writerow(list(row))