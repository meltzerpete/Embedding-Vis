import itertools
from neo4j.v1 import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "neo"))

# lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6]
# diffusions = [1, 2, 5, 10]

lambdas = [0.1, 0.4, 0.5]
diffusions = [2, 5, 10]

with driver.session() as session:
    for l, d in itertools.product(lambdas, diffusions):
        print("Lambda: " + str(l))
        props = {
            "lambda": l,
            "diffusions": d,
            "writeProperty": "embedding_{0}_{1}".format(l, d)
        }
        result = session.run("""\
        CALL algo.deepgl(null, null, 
          { pruningLambda: $lambda, 
            diffusions: $diffusions,
            writeProperty: $writeProperty
          }
        )
        """, props)

        for row in result:
            print(row)
