import pandas
from neo4j.v1 import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "neo"))

with driver.session() as session:
    result = session.run("""\
    match(p: Person)
    where
    p.label in [1, 4, 7, 14, 15, 21]
    WITH
    p.label
    AS
    label, p.partition as partition, count(*)
    AS
    count
    ORDER
    BY
    label, partition, count
    desc
    return label, partition, count
                             """)

    df = pandas.DataFrame(dict(row) for row in result)

accuracies = df.groupby("label")["count"].max() / df.groupby("label")["count"].sum()
mean_accuracy = accuracies.mean()
print("accuracies: ", accuracies)
print("mean:", mean_accuracy)