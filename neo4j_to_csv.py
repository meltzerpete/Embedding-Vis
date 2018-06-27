from neo4j.v1 import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "neo"))

with driver.session() as session:
#     result = session.run("""\
#     CALL db.propertyKeys()
#     YIELD propertyKey
#     WHERE propertyKey STARTS WITH "embedding_" AND not(propertyKey CONTAINS "lambda")
#     RETURN propertyKey
#     """)

    # property_keys = [row["propertyKey"] for row in result]
    property_keys = ["embedding"]

    # f1 f2 f3 f4 f5 .. fn label

    for property_key in property_keys:
        with open("{0}.txt".format(property_key), "w") as property_key_file:
            print("Property Key: ", property_key)
            result = session.run("""\
            MATCH (enzyme:SecondaryStructure)
            RETURN enzyme[$propertyKey] AS embedding, enzyme.label AS label
            order by label
            """, {"propertyKey": property_key})

            for row in result:
                property_key_file.write("{0} {1}\n".format(" ".join([str(item) for item in row["embedding"]]), row["label"]))
