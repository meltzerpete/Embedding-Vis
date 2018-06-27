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
    property_keys = ["embedding_lpa"]

    # f1 f2 f3 f4 f5 .. fn label

    for property_key in property_keys:
        with open("emails/{0}-emails.txt".format(property_key), "w") as property_key_file:
            print("Property Key: ", property_key)
            result = session.run("""\
            // MATCH (enzyme:Person)
            // RETURN enzyme[$propertyKey] AS embedding, enzyme.label AS label
            MATCH (n:Node) 
            WITH n.label as class, count(*) AS c
            ORDER BY c DESC
            WITH class WHERE c > 50
            WITH class ORDER BY class
            with collect(class) AS biggestClasses
            MATCH (p:Node) WHERE p.label IN biggestClasses
            RETURN p.`embedding-python` AS embedding, apoc.coll.indexOf(biggestClasses, p.label) AS label, p.label as initialLabel
            ORDER BY label
            """, {"propertyKey": property_key})

            for row in result:
                property_key_file.write(
                    "{0} {1}\n".format(" ".join([str(item) for item in row["embedding"]]), row["label"]))
