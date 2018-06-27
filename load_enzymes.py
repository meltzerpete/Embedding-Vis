from neo4j.v1 import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "neo"))

with driver.session() as session:
    session.run("""\
    LOAD CSV FROM "file:///ENZYME118.edgelist" AS row
    FIELDTERMINATOR " "
    MERGE (e1:SecondaryStructure {id: row[0]})
    MERGE (e2:SecondaryStructure {id: row[1]})
    MERGE (e1)-[:LINK]->(e2)""")

    session.run("""\
    LOAD CSV FROM "file:///all.labels" AS row
    FIELDTERMINATOR " "
    MATCH (e:SecondaryStructure {id: row[0]})
    SET  e.label =toInteger(row[1])-1
    """)

    session.run("""\
    load csv from "file:///ENZYME118.attributes"  as row
    FIELDTERMINATOR " "
    with toString(toInteger(row[0])) AS nodeId, row[1..] AS properties
    MATCH (s:SecondaryStructure {id: nodeId})
    WITH s, properties
    UNWIND range(0, size(properties)-1) AS index
    CALL apoc.create.setProperty(s, "property_" + index, toFloat(properties[index])) YIELD node
    return count(*)
    
    
    """)

