from neo4j import GraphDatabase
import src.utils as utils


def get_stat_queries():
    return {
        "node_labels_stats": "MATCH (n) RETURN distinct labels(n) as node_label, count(*) as count",
        "rel_labels_stats": "MATCH (n)-[r]->(n2) RETURN distinct type(r) as rel_label, count(*) as count",
        "triplet-type-list": "MATCH (x)-[r]->(y) RETURN distinct HEAD(labels(x)) as head, type(r), head(labels(y)) as tail"
    }


class DBManager():
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri=uri, auth=(username, password))
        self.uri = uri
        self.user = username
        self.pwd = password

    def get_mapping_queries(self, db):
        with self.driver.session(database=db) as session:
            node_types = session.run("MATCH(n) RETURN  DISTINCT labels(n)[0] as typen")  # getting all node types
            node_types = [record['typen'] for record in node_types]  # extracting data into a list
        mapping_queries = {node.lower(): f"MATCH (n:{node}) RETURN n.name as name" for node in
                           node_types}  # generating queries for node types
        mapping_queries['rel'] = """MATCH (n)-[r]-(n2)
            RETURN DISTINCT toLower(type(r)) as rel_label"""  # generating queries for edge types
        return mapping_queries

    def get_relation_queries(self, db):
        with self.driver.session(database=db) as session:
            triplets = session.run("""MATCH p=(a)-[r]->(b)
                    RETURN DISTINCT labels(a)[0] as source,
                        type(r) as relation,
                        labels(b)[0] as destination""")
            triplets = [(t['source'], t['relation'], t['destination']) for t in triplets]
        relation_queries = {str(tuple(map(lambda x: x.lower(),t))):
                                f"""MATCH (a:{t[0]})-[r:{t[1]}]->(b:{t[2]})
                                RETURN a.name as source_name, b.name as dest_name"""
                            for t in triplets}
        # relation_queries["('artwork', 'elicits', 'emotion')"] = """
        # match(a:Artwork)-[r]-(e:Emotion)
        # with a, sum(r.arousal) as sum_arousal, e
        # with a, max(sum_arousal) as max_arousal
        # match(a)-[r2]-(e2:Emotion)
        # with a, sum(r2.arousal) as sum2, e2, max_arousal
        # where sum2 = max_arousal
        # return a.name as source_name, collect(e2.name)[0] as dest_name
        # """
        return relation_queries

    def get_artworks(self, db):
        with self.driver.session(database=db) as session:
            query = 'match (a:Artwork) return a.name as name'
            ans = list(map(lambda x: x['name'], session.run(query)))
        return ans
