import itertools
from neo4j.v1 import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "neo"))

lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6]
diffusions = [1, 2, 5, 10]

with driver.session() as session:
    for l, d in itertools.product(lambdas, diffusions):
        print("Lambda: " + str(l))
        result = session.run("""\
        call algo.deepgl.stream(null, null, {pruningLambda: $lambda, diffusions: $diffusions})
        """, {"lambda": l, "diffusions": d})

        for row in result:
            print(row)
