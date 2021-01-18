import numpy as np


CLASS = 'Class'
EPS = np.finfo(float).eps


def calculate_dataset_entropy(df):
	entropy = 0
	values = df[CLASS].unique()

	for value in values:
		fraction = len(df[df[CLASS] == value]) / (len(df) + EPS)
		entropy += - fraction * np.log(fraction + EPS)

	return entropy


def calculate_attribute_entropy(df, attribute):
	att_values = df[attribute].unique()
	entropy = 0

	for value in att_values:
		att_df = df[df[attribute] == value]
		att_entropy = calculate_dataset_entropy(att_df)

		fraction = len(att_df) / (len(df) + EPS)
		entropy += - fraction * att_entropy

	return abs(entropy)


def find_winner(df):
	information_gains = [
		calculate_dataset_entropy(df) - calculate_attribute_entropy(df, attribute)
		for attribute in df.keys()[:-1]
	]

	return df.keys()[np.argmax(information_gains)]


def build_tree(df, tree=None):
	node = find_winner(df)
	attribute_values = np.unique(df[node])

	if tree is None:
		tree = {
			node: {}
		}

	for value in attribute_values:
		sub_table = df[df[node] == value].copy().reset_index(drop=True)
		class_value, counts = np.unique(sub_table[CLASS], return_counts=True)

		if len(counts) == 1:
			tree[node][value] = class_value[0]
		else:
			tree[node][value] = build_tree(sub_table)

	return tree
