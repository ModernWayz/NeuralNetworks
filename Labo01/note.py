import csv

train_x = []
train_y = []

test_x = []
test_y = []

with open('data/train.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        train_x.append(list(map(lambda x: float(x), row[1:])))
        train_y.append(float(row[0]))

with open('data/test.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        test_x.append(list(map(lambda x: float(x), row[1:])))
        test_y.append(float(row[0]))