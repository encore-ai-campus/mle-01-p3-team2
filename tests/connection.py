import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.environ["NEO4J_URI"]
username = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]

print("접속 URI:", uri)

driver = GraphDatabase.driver(
    uri,
    auth=(username, password),
    connection_timeout=10,
)

try:
    driver.verify_connectivity()
    print("Neo4j 연결 성공")
except Exception as e:
    print("연결 실패:", type(e).__name__, str(e))
finally:
    driver.close()