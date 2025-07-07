import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

@st.cache_data
def load_activity_data():
    
    uri = st.secrets["NEO4J_URI"]
    user = st.secrets["NEO4J_USER"]
    password = st.secrets["NEO4J_PASSWORD"]
    
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

@st.cache_data
def fetch_run_data():

    uri = st.secrets["NEO4J_URI"]
    user = st.secrets["NEO4J_USER"]
    password = st.secrets["NEO4J_PASSWORD"]
    
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def query_runs(tx):
        query = """
        MATCH (t:Tenant)<-[:BELONGS_TO]-(:User)-[:PERFORMED]->(:Action)-[:PROVISIONS]->(i:Instance)-[:HAS_RUN]->(r:Run)
        RETURN 
            t.name AS tenant,
            r.start_date AS start,
            r.end_date AS end,
            r.avg_cpu_usage_percent AS avg_cpu,
            i.id AS instance_id
        """
        result = tx.run(query)
        records = []
        for record in result:
            if record["start"] and record["end"]:
                records.append({
                    "tenant": record["tenant"],
                    "start": record["start"].to_native(),
                    "end": record["end"].to_native(),
                    "avg_cpu": float(record["avg_cpu"]) if record["avg_cpu"] is not None else 0.0,
                    "instance_id": record["instance_id"]
                })
        return pd.DataFrame(records)

    with driver.session() as session:
        df = session.execute_read(query_runs)
    driver.close()
    return df

@st.cache_data
def fetch_instance_counts():

    uri = st.secrets["NEO4J_URI"]
    user = st.secrets["NEO4J_USER"]
    password = st.secrets["NEO4J_PASSWORD"]

    driver = GraphDatabase.driver(uri, auth=(user, password))

    def query_instances(tx):
        query = """
        MATCH (t:Tenant)<-[:BELONGS_TO]-(:User)-[:PERFORMED]->(:Action)-[:PROVISIONS]->(i:Instance)
        RETURN 
            t.name AS tenant,
            i.id AS instance_id,
            i.instance_type AS instance_type
        """
        result = tx.run(query)
        return pd.DataFrame([dict(r) for r in result])

    with driver.session() as session:
        df = session.execute_read(query_instances)
    driver.close()
    return df
