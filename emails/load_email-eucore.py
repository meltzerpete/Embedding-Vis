from neo4j.v1 import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "neo"))

with driver.session() as session:
    session.run("""\
    LOAD CSV FROM "file:///emails.edgelist" AS row
    FIELDTERMINATOR " "
    MERGE (e1:Person {id: row[0]})
    MERGE (e2:Person {id: row[1]})
    MERGE (e1)-[:EMAILED]->(e2)""")

    session.run("""\
    LOAD CSV FROM "file:///emails.labels" AS row
    FIELDTERMINATOR " "
    MATCH (e:Person {id: row[0]})
    SET  e.label =toInteger(row[1])
    """)

