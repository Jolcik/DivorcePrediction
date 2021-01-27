import numpy as np


CLASS = 'Class'


def evaluate_row(df, tree):
	if isinstance(tree, np.int64):
		return tree

	atr_value = df[tree['attribute']]
	tree_value = tree['left'] if atr_value <= tree['split'] else tree['right']

	if isinstance(tree_value, dict):
		return evaluate_row(df, tree_value)

	return tree_value


def build_tree(df, max_depth=5, depth=1):
	if len(df) == 1 or depth >= max_depth:
		return df[CLASS].value_counts().idxmax()

	attributes = df.columns.tolist()[:-1]

	best_gini = 1
	best_split = -1
	best_atr = None

	for atr in attributes:
		gini, split = get_best_split(df, atr)

		if gini is not None and gini < best_gini:
			best_atr, best_gini, best_split = atr, gini, split

	if best_atr:
		return {
			'attribute': best_atr,
			'split': best_split,
			'left': build_tree(df[df[best_atr] <= best_split].drop(columns=[best_atr]), max_depth, depth+1),
			'right': build_tree(df[df[best_atr] > best_split].drop(columns=[best_atr]), max_depth, depth+1),
		}

	return df[CLASS].value_counts().idxmax()


def get_best_split(df, attribute):
	attribute_values = sorted(df[attribute].unique())

	parent_gini = calculate_gini_index(df)

	best_gini = 1
	best_split = -1

	for value in sorted(attribute_values)[:-1]:
		left_df = df[df[attribute] <= value]
		left_gini = calculate_gini_index(left_df)

		right_df = df[df[attribute] > value]
		right_gini = calculate_gini_index(right_df)

		child_gini = (len(left_df) * left_gini + len(right_df) * right_gini) / len(df)

		if child_gini < best_gini:
			best_gini = child_gini
			best_split = value

	if best_gini < parent_gini:
		return best_gini, best_split

	return None, None


def calculate_gini_index(df):
	_, counts = np.unique(df[CLASS], return_counts=True)
	return 1 - sum(
		np.square(counts / len(df))
	)
