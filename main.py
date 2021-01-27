import pandas as pd
from src import id3, cart
from calculations import test_forest


data = pd.read_csv('data/divorce.csv', delimiter=';')


# these functions were created to provide a way to universally test CART tree building
# with different max depth parameter value
def cart_build_tree_3(df, depth=1):
	return cart.build_tree(df, max_depth=3, depth=depth)


def cart_build_tree_5(df, depth=1):
	return cart.build_tree(df, max_depth=5, depth=depth)


def cart_build_tree_8(df, depth=1):
	return cart.build_tree(df, max_depth=8, depth=depth)


def cart_build_tree_10(df, depth=1):
	return cart.build_tree(df, max_depth=10, depth=depth)


function_cases = [
	(id3.build_tree, id3.evaluate_row, 'id3'),
	(cart_build_tree_3, cart.evaluate_row, 'cart3'),
	(cart_build_tree_5, cart.evaluate_row, 'cart5'),
	(cart_build_tree_8, cart.evaluate_row, 'cart8'),
	(cart_build_tree_10, cart.evaluate_row, 'cart10'),
]

cross_validation_factors = [2, 5, 10]
tree_numbers = [2, 5, 10, 20, 30, 40, 50, 100]
attribute_numbers = [2, 5, 10, 15, 20, 30, 40]


for build, evaluate, name in function_cases:
	test_forest(
		data=data,
		cross_validation_factors=cross_validation_factors,
		build_tree=build,
		evaluate_row=evaluate,
		tree_numbers=tree_numbers,
		attribute_numbers=attribute_numbers,
		repetitions=3,
		name=name,
	)
