import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
@st.cache_data
def load_activity_data():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def fetch_data(tx):
        query = """
        MATCH (u:User)-[:BELONGS_TO]->(t:Tenant)
        MATCH (u)-[:PERFORMED]->(a:Action)
        OPTIONAL MATCH (a)-[:PROVISIONS]->(i:Instance)
        RETURN 
            t.name AS tenant, 
            u.username AS username, 
            a.type AS type, 
            a.ts AS ts, 
            i.name AS instance_name,
            i.id AS instance_id,
            a.message AS message, 
            a.objecttype AS objecttype, 
            a.id AS id
        ORDER BY a.ts DESC
        """

        result = tx.run(query)
        records = []

        for record in result:
            row = dict(record)
            if "ts" in row and hasattr(row["ts"], "to_native"):
                row["ts"] = row["ts"].to_native()
            records.append(row)

        return pd.DataFrame(records)

    with driver.session() as session:
        df = session.execute_read(fetch_data)

    driver.close()
    return df
