import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import json
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = "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"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

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

@st.cache_data
def fetch_run_data():

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
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

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

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


@st.cache_data
def fetch_execution_data():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def query_executions(tx):
        query = """
        MATCH (u:User)-[:BELONGS_TO]->(t:Tenant)
        MATCH (u)-[:CREATED]->(j:Job)-[:HAS_EXECUTION]->(e:Execution)
        RETURN 
            t.name AS tenant,
            u.username AS username,
            j.name AS job_name,
            e.startDate AS ts,
            e.status AS status,
            e.duration AS duration,
            e.type AS process_type
        ORDER BY e.startDate DESC
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
        df = session.execute_read(query_executions)
    driver.close()
    return df

@st.cache_data
def fetch_temporal_activity_data(selected_tenants=None):
    from neo4j import GraphDatabase
    import os

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def run_query(tx):
        query = """
        MATCH (t:Tenant)<-[:BELONGS_TO]-(u:User)
        OPTIONAL MATCH (u)-[:PERFORMED]->(a:Action)
        OPTIONAL MATCH (u)-[:CREATED]->(j:Job)-[:HAS_EXECUTION]->(e:Execution)
        WHERE $tenants IS NULL OR t.name IN $tenants
        RETURN 
            t.name AS tenant,
            u.username AS user,
            a.ts AS action_ts,
            e.startDate AS exec_start,
            e.duration AS exec_duration,
            e.status AS exec_status,
            e.type AS exec_type
        ORDER BY t.name, u.username, action_ts, exec_start
        """
        result = tx.run(query, tenants=selected_tenants)
        return [dict(r) for r in result]

    with driver.session() as session:
        data = session.execute_read(run_query)

    driver.close()
    return pd.DataFrame(data)


def prepare_llm_friendly_json(df):
    grouped = df.groupby(["tenant", "user"])
    tenant_map = {}

    for (tenant, user), group in grouped:
        entry = tenant_map.setdefault(tenant, {"tenant": tenant, "users": []})
        activity = []

        for _, row in group.iterrows():
            if pd.notna(row.get("action_ts")):
                activity.append({
                    "type": "action",
                    "ts": row["action_ts"].isoformat()
                })
            if pd.notna(row.get("exec_start")):
                activity.append({
                    "type": "execution",
                    "start": row["exec_start"].isoformat(),
                    "duration": row["exec_duration"],
                    "status": row["exec_status"],
                    "exec_type": row["exec_type"]
                })

        entry["users"].append({
            "user": user,
            "activity": activity
        })

    return tenant_map

def get_temporal_insights_from_ai(payload_json):
    try:
        prompt = (
            "Analyze the following tenant activity logs which include timestamped actions and job executions.\n"
            "Look for hidden patterns, anomalies (e.g., failed jobs, usage gaps), night-time activity, or inefficiencies.\n\n"
            f"{payload_json}"
        )

        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Failed to fetch insights: {e}"

