import random


def evaluate(df, forest, evaluate_tree):
	all_results = {}

	for tree in forest:
		result = evaluate_tree(tree, df)

		if result not in all_results.keys():
			all_results[result] = 1
		else:
			all_results[result] += 1

	max_result = list(all_results.keys())[0]
	max_count = all_results[max_result]

	for result in list(all_results.keys()):
		if all_results[result] > max_count:
			max_result = result
			max_count = all_results[result]

	return max_result


def build(df, build_tree, trees_number=50, attributes_number=10):
	trees = []

	for _ in range(trees_number):
		# it's a property of random forest algorithm to select random rows with replacement
		randomized_df = df.sample(trees_number, replace=True).reset_index(drop=True)

		attributes = df.columns.tolist()
		randomized_attributes = random.sample(attributes[:-1], attributes_number)

		# add Class column, we add it here because it shouldn't be included in randomizing of attributes
		randomized_attributes.append(attributes[-1])

		trees.append(build_tree(randomized_df[randomized_attributes]))

	return trees
