import pandas as pd
import numpy as np

all_adjacencies = pd.read_csv("ENZYMES/ENZYMES_A.txt", ",", names=["row", "col"])
all_graph_indicator = pd.read_csv("ENZYMES/ENZYMES_graph_indicator.txt", " ", names=["graph"])
all_node_labels = pd.read_csv("ENZYMES/ENZYMES_node_labels.txt", " ", names=["label"])
all_node_attributes = pd.read_csv("ENZYMES/ENZYMES_node_attributes.txt", ",", header=None)

all_adjacencies_with_graph_id = pd.merge(all_adjacencies, all_graph_indicator, left_on="row", right_index=True)

for id in [118, 295, 296]:
    current = all_adjacencies_with_graph_id[all_adjacencies_with_graph_id.graph == id][["row", "col"]].values
    np.savetxt("ENZYME" + str(id) + ".edgelist", current, fmt="%d", delimiter=" ")
    nodes = all_graph_indicator[all_graph_indicator.graph == id]
    labels = nodes.merge(all_node_labels, left_index=True, right_index=True)["label"].reset_index().values
    np.savetxt("ENZYME" + str(id) + ".labels", labels, fmt="%d", delimiter=" ")
    # attributes
    attributes = all_node_attributes.iloc[nodes.index].reset_index()
    np.savetxt("ENZYME" + str(id) + ".attributes", attributes, fmt="%f", delimiter=" ")

all_node_labels["id"] = list(np.arange(1, all_node_labels.size + 1))
np.savetxt("all.labels", all_node_labels[["id", "label"]], fmt="%d", delimiter=" ")

