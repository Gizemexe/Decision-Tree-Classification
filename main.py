import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import matplotlib
matplotlib.use('TkAgg')  # Alternative visualization backend

# Loading training and test data
train_data = pd.read_excel("trainDATA.xlsx")
test_data = pd.read_excel("testDATA.xlsx")

# Encoding categorical data
def encode_data(df):
    return df.apply(lambda col: pd.Categorical(col).codes if col.dtypes == 'object' else col)

train_data = encode_data(train_data)
test_data = encode_data(test_data)

# Separating training and test data into input(X) and output(y)
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :]

# Gini index calculation
def gini_index(groups, classes):
    total_samples = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:  # Empty group check
            continue
        score = 0.0
        for class_val in classes:
            proportion = np.sum(group[:, -1] == class_val) / size
            score += proportion ** 2
        gini += (1.0 - score) * (size / total_samples)
    return gini

# Splitting data into two groups based on a property and its value
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return np.array(left), np.array(right)

# Creating the Decision Tree
column_names = ["Price", "MaintPrice", "NoofDoors", "Persons", "Lug_size", "Safety"]

def build_tree(dataset, depth=0, max_depth=5):
    if len(np.unique(dataset[:, -1])) == 1 or depth == max_depth:
        return {'type': 'leaf', 'class': np.unique(dataset[:, -1])[0], 'gini': 0.0, 'samples': len(dataset)}

    best_index, best_value, best_score, best_groups = None, None, float('inf'), None
    classes = np.unique(dataset[:, -1])
    for index in range(dataset.shape[1] - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, classes)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups

    if best_groups is None or best_index is None:
        return {'type': 'leaf', 'class': np.unique(dataset[:, -1])[0], 'gini': best_score, 'samples': len(dataset)}

    left = build_tree(best_groups[0], depth + 1, max_depth)
    right = build_tree(best_groups[1], depth + 1, max_depth)

    return {
        'type': 'node',
        'feature': column_names[best_index],
        'value': best_value,
        'gini': best_score,
        'samples': len(dataset),
        'left': left,
        'right': right
    }

# Make predictions on the decision tree
def predict(tree, row):
    # If at a leaf node, return the class
    if tree['type'] == 'leaf':
        return tree['class']

    # At a decision node, check the feature
    feature_name = tree['feature']
    feature_index = column_names.index(feature_name)

    # Follow left or right branch
    if row[feature_index] < tree['value']:
        return predict(tree['left'], row)
    else:
        return predict(tree['right'], row)

# Combine training data into a single dataset
dataset = np.array(pd.concat([X_train, y_train], axis=1))

# Train the Decision Tree
tree = build_tree(dataset, max_depth=10)

# Classification of Test Data
test_predictions = [predict(tree, row) for row in np.array(X_test)]

# Save results
output = pd.DataFrame({"Predictions": test_predictions})
output.to_excel("classification_results.xlsx", index=False)

# Tree Visualization
# Visualizing the tree with NetworkX
def plot_tree(tree, graph=None, node_id=0, parent_id=None, edge_label=None, pos=None, x_offset=1,
              y_offset=1, layer=1):
    if graph is None:
        graph = nx.DiGraph()
        pos = {}

    if tree['type'] == 'leaf':
        node_label = f"Leaf: {tree['class']}\nGini: {tree['gini']:.2f}\nSamples: {tree['samples']}"
    else:
        node_label = f"{tree['feature']} <= {tree['value']}\nGini: {tree['gini']:.2f}\nSamples: {tree['samples']}"

    graph.add_node(node_id, label=node_label)
    pos[node_id] = (x_offset, -layer)

    if parent_id is not None:
        graph.add_edge(parent_id, node_id, label=edge_label)

    if tree['type'] != 'leaf':
        graph, pos = plot_tree(
            tree['left'], graph, node_id=node_id * 2 + 1, parent_id=node_id,
            edge_label="True", pos=pos, x_offset=x_offset - 1 / layer,
            y_offset=y_offset, layer=layer + 1
        )
        graph, pos = plot_tree(
            tree['right'], graph, node_id=node_id * 2 + 2, parent_id=node_id,
            edge_label="False", pos=pos, x_offset=x_offset + 1 / layer,
            y_offset=y_offset, layer=layer + 1
        )

    return graph, pos

# Visualization of decision tree
graph, pos = plot_tree(tree)

# Draw the tree
def draw_tree(graph, pos):
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=False, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
    node_labels = nx.get_node_attributes(graph, 'label')
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.show()

draw_tree(graph, pos)
