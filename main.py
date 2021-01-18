from pprint import pprint
import pandas as pd
from src.id3 import build_tree


def get_result_for_tree(tree, df):
    att = list(tree.keys())[0]
    df_att = df[att]

    if df_att not in tree[att]:
        return 0

    if tree[att][df_att] is dict:
        return get_result_for_tree(tree[att][df_att], df)

    return tree[att][df_att]


data = (pd.read_csv('data/divorce.csv', delimiter=';')
        .sample(frac=1)
        .reset_index(drop=True)
        .copy())

results = {}
for fraction in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    sum = 0
    for i in range(50):
        delimiter = int(len(data) * fraction)
        training_data = data[:delimiter]
        test_data = data[delimiter:]

        tree = build_tree(training_data)
        # pprint(tree)

        success = 0
        for _, row in test_data.iterrows():
            real = row.iloc[-1]
            forecasted = get_result_for_tree(tree, row)

            if real == forecasted:
                success += 1

        sum += success / len(test_data)

    results[fraction] = sum / 50

print(results)
