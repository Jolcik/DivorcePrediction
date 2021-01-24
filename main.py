from pprint import pprint
import pandas as pd
from src.id3 import build_tree, evaluate_row_tree
from src import random_forest


data = (pd.read_csv('data/divorce.csv', delimiter=';')
        .sample(frac=1)
        .reset_index(drop=True)
        .copy())


row_count = len(data)
k = 5

all_all_success = 0
all_all_count = 0

for j in range(10):
    all_success = 0
    all_count = 0

    for i in range(k):
        start = i * row_count // k
        end = start + row_count // k

        test_data = data[start:end].copy()
        training_data = data[:start].append(data[end:]).copy()

        forest = random_forest.build(training_data, build_tree, trees_number=150, attributes_number=5)

        success = 0
        count = 0
        for _, row in test_data.iterrows():
            result = random_forest.evaluate(row, forest, evaluate_row_tree)
            expected = row.iloc[-1]

            if result != -1:
                count += 1
                if result == expected:
                    success += 1

        all_success += success
        all_count += count

        print(f'        {i+1}. SUCCESS RATE: {(success / count) * 100:.2f}%')

    all_all_success += all_success
    all_all_count += all_count
    print(f'    {j+1}. CUMULATIVE SUCCESS RATE: {(all_success / all_count) * 100:.2f}%')

print(f'WHOLE SUCCESS RATE: {(all_all_success / all_all_count) * 100:.2f}%')
