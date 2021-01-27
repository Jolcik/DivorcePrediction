import pandas as pd

from src import random_forest


def test_forest(data, cross_validation_factors, build_tree, evaluate_row, tree_numbers, attribute_numbers, repetitions, name):
	for rep in range(repetitions):
		df = data.copy().sample(frac=1).reset_index(drop=True)
		result_df = pd.DataFrame(columns=['cross', 'section', 'trees', 'attributes', 'success', 'count', 'result'])

		for cr_factor in cross_validation_factors:
			for trees in tree_numbers:
				for attributes in attribute_numbers:

					all_success = 0
					all_count = 0

					for i in range(cr_factor):
						start = i * len(df) // cr_factor
						end = start + len(df) // cr_factor

						test_data = df[start:end].copy()
						training_data = df[:start].append(data[end:]).copy()

						forest = random_forest.build(training_data, build_tree, trees_number=trees, attributes_number=attributes)

						success = 0
						count = 0
						for _, row in test_data.iterrows():
							result = random_forest.evaluate(row, forest, evaluate_row)
							expected = row.iloc[-1]

							if result != -1:
								count += 1
								if result == expected:
									success += 1

						all_success += success
						all_count += count

						if count == 0:
							count = 1

						print(f'        {i + 1}. SUCCESS RATE: {(success / count) * 100:.2f}%')

						# result_df.loc[result_df.shape[0]] = [cr_factor, i+1, trees, attributes, success, count, (success / count) * 100]

					print(f'CUMULATIVE SUCCESS RATE: {(all_success / all_count) * 100:.2f}%')

					result_df.loc[result_df.shape[0]] = [cr_factor, 0, trees, attributes, all_success, all_count, (all_success / all_count) * 100]

		result_df.to_csv(f'{name}_{rep}.csv')
