import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
from neo4j.v1 import GraphDatabase
import tensorflow as tf


# load data
from sklearn.preprocessing import StandardScaler


def load_data(driver, edgelistFile, labelsFile, attributesFile=None):
    print("loading data")
    with driver.session() as session:
        session.run("""\
            LOAD CSV FROM "file:///" + $edgelistFile AS row
            FIELDTERMINATOR " "
            MERGE (e1:Node {id: row[0]})
            MERGE (e2:Node {id: row[1]})
            MERGE (e1)-[:LINK]->(e2)
            """, {"edgelistFile": edgelistFile})

        session.run("""\
            LOAD CSV FROM "file:///" + $labelsFile AS row
            FIELDTERMINATOR " "
            MATCH (e:Node {id: row[0]})
            SET  e.label =toInteger(row[1])-1
            """, {"labelsFile": labelsFile})

        if attributesFile is not None:
            session.run("""\
                load csv from "file:///" + $attributesFile  as row
                FIELDTERMINATOR " "
                with toString(toInteger(row[0])) AS nodeId, row[1..] AS properties
                MATCH (s:Node {id: nodeId})
                WITH s, properties
                UNWIND range(0, size(properties)-1) AS index
                CALL apoc.create.setProperty(s, "property_" + index, toFloat(properties[index])) YIELD node
                return count(*)
                """, {"attributesFile": attributesFile})


# run algo
def run_algo(driver, write_property, node_features=[], pruningLambda=0.6, diffusions=3, iterations=3):
    print("running algo")
    with driver.session() as session:
        params = {
            "writeProperty": write_property,
            "nodeFeatures": node_features,
            "pruningLambda": pruningLambda,
            "diffusions": diffusions,
            "iterations": iterations
        }
        result = session.run("""
        call
        algo.deepgl(
            null,
            null,
            {nodeFeatures: $nodeFeatures,
             pruningLambda: $pruningLambda,
             diffusions: $diffusions,
             iterations: $iterations,
             writeProperty: $writeProperty})
        """, params)
        print(result.peek())


# extract results
def extract_results(driver, property_key):
    print("extracting results")
    with driver.session() as session:
        result = session.run("""\
        MATCH (n) 
        WITH n.label as class, count(*) AS c
        ORDER BY c DESC
        WITH class WHERE c > 50
        WITH class ORDER BY class
        with collect(class) AS biggestClasses
        MATCH (p:Node) WHERE p.label IN biggestClasses
        RETURN p.`%s` AS embedding, apoc.coll.indexOf(biggestClasses, p.label) AS label, p.label as initialLabel
        ORDER BY label
        """ % property_key)

        df = pd.DataFrame(dict(row) for row in result)

    emb = df["embedding"].apply(pd.Series).values
    labels = df["label"].values
    return emb, labels


# visualise
def visualise(emb, labels):
    print("visualising results")
    # Heatmap
    colours = ['r', 'g', 'b', 'black', 'y', 'orange']
    cols = pd.DataFrame(labels).apply(lambda x: colours[int(x)], axis=1).values

    dist = np.ndarray([len(emb), len(emb)])

    for i, e1 in enumerate(emb):
        for j, e2 in enumerate(emb):
            dist.itemset((i, j), np.linalg.norm(e1 - e2, 2))

    plt.imshow(dist)
    plt.axes().xaxis.tick_top()
    plt.xticks(np.arange(len(dist)), labels)
    plt.yticks(np.arange(len(dist)), labels)
    plt.show()

    # 2D Visualisation
    # from: https://baoilleach.blogspot.com/2014/01/convert-distance-matrix-to-2d.html
    adist = dist
    amax = np.amax(adist)
    adist /= amax

    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    results = mds.fit(adist)

    coords = results.embedding_

    plt.subplots_adjust(bottom=0.1)
    plt.scatter(
        coords[:, 0], coords[:, 1], marker='o', c=cols
    )
    # for label, x, y in zip(labels, coords[:, 0], coords[:, 1]):
    #     plt.annotate(
    #         label,
    #         xy = (x, y), xytext = (-20, 20),
    #         textcoords = 'offset points', ha = 'right', va = 'bottom',
    #         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    #         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    plt.show()


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    print(dict(features))
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# evaluate (classifier)
def evaluate(emb, labels):
    print("evaluating with classifier")
    data = pd.DataFrame(emb)

    data.columns = [str(col) for col in data.columns.get_values()]

    data["label"] = labels

    train_index = int(len(data) * 0.6)
    train = data[:train_index]
    test = data[train_index:]

    train_x = train.drop("label", axis=1)
    train_y = train["label"]

    test_x = test.drop("label", axis=1)
    test_y = test["label"]

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[30, 30],
        # optimizer=
        # activation_fn=tf.nn.softmax,
        # The model must choose between 3 classes.
        n_classes=data["label"].unique().size)

    # Train the Model.
    classifier.train(
        input_fn=lambda: train_input_fn(train_x, train_y, data.shape[0]),
        steps=1000)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(test_x, test_y, data.shape[0]))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


def evaluate_sk(emb, labels):
    print("evaluating with classifier")
    X = pd.DataFrame(emb)

    # X.columns = [str(col) for col in X.columns.get_values()]

    y = labels

    X = StandardScaler().fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.4, random_state=42)

    # print("train_x ", train_x)
    # print("train_y ", train_y)

    clf = MLPClassifier(solver='sgd',
                        activation='tanh',
                        learning_rate_init=0.001,
                        alpha=1e-5,
                        hidden_layer_sizes=(30, 30),
                        max_iter=10000,
                        batch_size=X.shape[0],
                        random_state=0)

    clf.n_outputs_ = 6
    clf.out_activation_ = "softmax"
    print(clf.get_params())
    clf.fit(train_x, train_y)

    mean_acc = clf.score(test_x, test_y)
    print(mean_acc)


if __name__ == "__main__":
    # execute only if run as a script
    driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "neo"))

    # load_data(driver, "emails.edgelist", "emails.labels")
    embedding_property_name = "embedding-python"
    # run_algo(driver, embedding_property_name, iterations=3)

    emb, labels = extract_results(driver, embedding_property_name)
    evaluate_sk(emb, labels)
    # visualise(emb, labels)
