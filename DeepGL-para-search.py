# coding: utf-8

# # DeepGL - Deep Feature Learning for graphs
# 
# We've implemented the [DeepGL](https://arxiv.org/abs/1704.08829) algorithm as a Neo4j procedure and this notebook shows our experiments with it against a SNAP email dataset.
# 
# First up let's import some things...

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
from neo4j.v1 import GraphDatabase
import time
from datetime import datetime
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import signal

driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "neo"))


def load_data(edge_list_file, labels_file, attributes_file):
    with driver.session() as session:
        session.run("MATCH(n) DETACH DELETE n")
        # session.run("DETACH DELETE n")

        session.run("CREATE CONSTRAINT ON (n:Node) ASSERT n.id IS UNIQUE")

        result = session.run("""            LOAD CSV FROM $edgelistFile AS row
            FIELDTERMINATOR " "
            MERGE (e1:Node {id: row[0]})
            MERGE (e2:Node {id: row[1]})
            MERGE (e1)-[:LINK]->(e2)
            """, {"edgelistFile": edge_list_file})
        print(result.summary().counters)

        result = session.run("""            LOAD CSV FROM $labelsFile AS row
            FIELDTERMINATOR " "
            MATCH (e:Node {id: row[0]})
            SET  e.label = toInteger(row[1])-1
            """, {"labelsFile": labels_file})
        print(result.summary().counters)

        df = pd.DataFrame([row.values() for row in result])

        # print('\n --- after loading row.values() df.head() --- \n',df.head())

        if attributes_file is not None:
            result = session.run("""                load csv from $attributesFile  as row
                FIELDTERMINATOR " "
                with toString(toInteger(row[0])) AS nodeId, row[1..] AS properties
                MATCH (s:Node {id: nodeId})
                WITH s, properties
                UNWIND range(0, size(properties)-1) AS index
                CALL apoc.create.setProperty(s, "property_" + index, toFloat(properties[index])) YIELD node
                return count(*)
                """, {"attributesFile": attributes_file})
            print(result.summary().counters)


def train_model(params):
    with driver.session() as session:
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
        result = session.run("MATCH(n) RETURN n LIMIT 20")

        df = pd.DataFrame(dict(row) for row in result)

        # print('\n --- after training df.head() --- ',df.head())
        print('result.peek: ', result.peek())


def get_result(embedding_property_name):
    with driver.session() as session:
        # result = session.run("MATCH(n) RETURN n LIMIT 20")
        # df = pd.DataFrame(dict(row) for row in result)
        # print('\n --- df.head() --- \n',df.head())

        result = session.run("""        
        MATCH (n)
        RETURN n.`%s` AS embedding, n.label AS label
        """ % embedding_property_name)

        df = pd.DataFrame(dict(row) for row in result)
        # print('\n --- df.head() in get result--- \n',df.head())
        emb = df["embedding"].apply(pd.Series).values
        labels = df["label"].values
    return emb, labels


def to_figure(emb, labels, embedding_property_name):
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
    plt.savefig('./out_fig/heat_map_1_{}.png'.format(embedding_property_name))

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

    # plt.show()

    plt.savefig('./out_fig/heat_map_2_{}.png'.format(embedding_property_name))


def mean_acc(emb, labels):
    global embedding_property_name
    X = pd.DataFrame(emb)
    y = labels
    X = StandardScaler().fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.33, random_state=42)
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
    # print("clf.get_params()")
    # print(clf.get_params())
    clf.fit(train_x, train_y)

    mean_acc = clf.score(test_x, test_y)
    print("mean_acc ", embedding_property_name, mean_acc)


def micro_f1(emb, labels):
    global embedding_property_name
    X = pd.DataFrame(emb)
    y = labels
    X = StandardScaler().fit_transform(X)

    f1_list = []
    for i in range(10):
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.33)
        lr_model = SGDClassifier(loss="log", penalty="l2")
        lr_model.fit(train_x, train_y)
        y_pred = lr_model.predict(test_x)
        f1 = f1_score(y_pred=y_pred, y_true=test_y, average='micro')
        f1_list.append(f1)
    f1_ave = np.mean(f1_list)
    f1_std = np.std(f1_list)

    print('f1', embedding_property_name, f1_ave, '+-', f1_std)
    return (f1_ave, f1_std)


def main(edge_list_file, labels_file, attributes_file,
         embedding_property_name, node_features, pruning_lambda, diffusions, iterations):
    print(datetime.now())
    t0 = time.time()
    params = {
        "writeProperty": embedding_property_name,
        "nodeFeatures": node_features,
        "pruningLambda": pruning_lambda,
        "diffusions": diffusions,
        "iterations": iterations
    }

    train_model(params)
    print('train_time ', embedding_property_name, time.time() - t0)
    emb, labels = get_result(embedding_property_name)
    print('get_result', time.time() - t0)

    mean_acc(emb, labels)
    # print ('mean acc', time.time()-t0)

    ave_std = micro_f1(emb, labels)
    # print ('f1', time.time()-t0)
    return ave_std


names = ['emails', 'ENZYMES', 'DD', 'NCI1', 'Flickr', 'YouTube']
edge_urls = {}
edge_urls[names[0]] = "https://github.com/meltzerpete/Embedding-Vis/raw/master/emails/emails.edgelist"
edge_urls[names[1]] = "https://raw.githubusercontent.com/koyamabraintree/dataset/master/ENZYMES/edges"
edge_urls[names[2]] = "https://raw.githubusercontent.com/koyamabraintree/dataset/master/DD/edges"
edge_urls[names[3]] = "https://raw.githubusercontent.com/koyamabraintree/dataset/master/NCI1/edges"
edge_urls[names[4]] = "https://raw.githubusercontent.com/koyamabraintree/dataset/master/Flickr/edges"
edge_urls[names[5]] = "https://raw.githubusercontent.com/koyamabraintree/dataset/master/YouTube/edges"

labels_urls = {}
labels_urls[names[0]] = "https://github.com/meltzerpete/Embedding-Vis/raw/master/emails/emails.labels"
labels_urls[names[1]] = "https://raw.githubusercontent.com/koyamabraintree/dataset/master/ENZYMES/node_labels"
labels_urls[names[2]] = "https://raw.githubusercontent.com/koyamabraintree/dataset/master/DD/node_labels"
labels_urls[names[3]] = "https://raw.githubusercontent.com/koyamabraintree/dataset/master/NCI1/node_labels"
labels_urls[names[4]] = "https://raw.githubusercontent.com/koyamabraintree/dataset/master/Flickr/node_labels"
labels_urls[names[5]] = "https://raw.githubusercontent.com/koyamabraintree/dataset/master/YouTube/node_labels"


# timeout to stop the programme
def signal_handler(signum, frame):
    raise Exception("Timed out of signal_handler")


signal.signal(signal.SIGALRM, signal_handler)

list_pruning_lambda = [0.5, 0.6, 0.7]
list_diffusions = [2, 3, 4]
list_iterations = [3, 4, 5, 6, 7]

finals = {}
for name in ['YouTube']:
    attributes_file = None
    edge_list_file = edge_urls[name]
    labels_file = labels_urls[name]
    node_features = []
    t0 = time.time()
    load_data(edge_list_file, labels_file, attributes_file)
    print('load_data', time.time() - t0)
    for pruning_lambda in list_pruning_lambda:
        for iterations in list_iterations:
            for diffusions in list_diffusions:
                signal.alarm(1800)
                attributes_file = None
                embedding_property_name = "embedding-python" + str(name) + '_pl{}_d{}_i{}'.format(
                    str(pruning_lambda).split('.')[-1], diffusions, iterations)
                try:
                    print('\n\n---- ', str(embedding_property_name), '-----\n')
                    print(pruning_lambda, diffusions, iterations)
                    main(edge_list_file, labels_file, attributes_file, embedding_property_name, node_features,
                         pruning_lambda, diffusions, iterations)
                except Exception as e:
                    print('except error: ', e)
                    finals[embedding_property_name] = 'Error'