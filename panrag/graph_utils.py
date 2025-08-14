from typing import List
from neo4j import GraphDatabase

from . import config


def graph_enabled() -> bool:
    return config.GRAPH_ENABLED


def get_driver():
    return GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))


def expand_neighbors(initial_docs: List):
    if not graph_enabled() or not initial_docs:
        return []
    ids = []
    for d in initial_docs:
        cid = d.metadata.get("chunk_id")
        if cid:
            ids.append(cid)
    if not ids:
        return []
    neighbors = []
    with get_driver().session() as session:
        for cid in ids:
            result = session.run(
                """
                MATCH (c:Chunk {id: $cid})
                OPTIONAL MATCH (c)-[:NEXT]->(n)
                OPTIONAL MATCH (p)-[:NEXT]->(c)
                RETURN collect(distinct n.id) + collect(distinct p.id) AS ids
                """,
                cid=cid,
            )
            row = result.single()
            if not row:
                continue
            for nid in (row["ids"] or []):
                if nid:
                    neighbors.append(nid)
    return neighbors
