from neo4j.v1 import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "neo"))

with driver.session() as session:
    session.run("""\
    LOAD CSV FROM "file:///ENZYME118.edgelist" AS row
    FIELDTERMINATOR " "
    MERGE (e1:Enzyme {id: row[0]})
    MERGE (e2:Enzyme {id: row[1]})
    MERGE (e1)-[:LINK]->(e2)""")

    session.run("""\
    LOAD CSV FROM "file:///ENZYME118.labels" AS row
    FIELDTERMINATOR " "
    MATCH (e:Enzyme {id: row[0]})
    SET  e.label =toInteger(row[1])-1
    """)
