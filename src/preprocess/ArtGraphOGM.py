from neo4j import GraphDatabase
import pandas
from pandas import DataFrame
import os
from sklearn.model_selection import train_test_split
from ast import literal_eval
import logging
import copy

logging.basicConfig(level=logging.INFO)
       
class ArtGraphDBConnector:    
    """
    The interface to connect with Neo4j DBMS and perform queries to the ArtGraph database.
    For starting you need to privide all the required information for creating the driver object,
    which will comunicate with the database.

    Args:
        uri (string): the connection uri
        username (string): neo4j username
        password (string): neo4j password
        database (string): the name you set when you imported the artgraph database
    """
    
    def __init__(self, uri, username, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
    
    def close(self):
        self.driver.close()
    
    def df_from_query(self, query):
        """Return the result of a query as a dataframe.

            Args:
                query (string): the query to be performed
            
            Returns:
                df_result (pandas.DataFrame): the result of the query
            
        """
        result = self.__run_query(query)
        df_result = DataFrame(result)
        return df_result

    def __run_query(self, query):
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.data()
            
class Relation():
    """A relation type.

    It holds relation's information, all edges of the relation in the graph and their optional attributes.
    It has also the capability of saving itself on disk, according to the OGB convention.
    """

    def __init__(self, id, name, source, destination):
        self.id = id
        self.name = name
        self.source = source
        self.destination = destination
        self._edges = None
        self._attributes = None
    
    def add_edges(self, edges):
        self._edges = edges

    def add_attributes(self, attributes):
        assert self._edges is not None
        assert len(self._edges) == len(attributes)

        self._attributes = attributes
    
    def get_edges(self):
        return self._edges
    
    def get_attributes(self):
        return self._attributes

    @property
    def edges_number(self):
        return len(self._edges)

    @property
    def relation_id(self):
        return DataFrame().from_dict({"edge": [self.id] * self.edges_number})

    def save_on_disk(self, root):
        folder_path = os.path.join(root, self.source + "___" + self.name + "___" + self.destination)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        logging.info(f'Saving on the disk relation {self.name} in path {folder_path}')

        self._edges.to_csv(os.path.join(folder_path, "edge.csv"), index=False, header=False)
        self.relation_id.to_csv(os.path.join(folder_path, "edge_reltype.csv"), index=False, header=False)
        with open(os.path.join(folder_path, "num-edge-list.csv"), 'w') as f:
            f.write(str(self.edges_number))

        if self._attributes is not None:
            self._attributes.to_csv(os.path.join(folder_path, "attributes.csv"), index=False, header=False)
        f.close()

class Relations():
    """Relation types of the graph.

    It contains all the relations of the graph and their relative information in a dictionary data structure.
    Each relation is a object of the call raw_generation.Relation.
    """

    def __init__(self):
        self._relations = {}
    
    def add_relation(self, name, relations):
        self._relations[name] = relations

    def get_relation(self, name):
        return self._relations[name]

    def __iter__(self):
        return iter(self._relations.items())
    
    def save_on_disk(self, root):
        logging.info('Saving on the disk all relations')
        for _, relation in self:
            relation.save_on_disk(root)

    def remove_relations(self, excluded_relation = [], valid_nodes = [], test_nodes = []):
        """Remove all relations which involve test artworks.
        Remove all relations which involve validation artwork expect relations that correspond to labels.

        Artwors are only into the head of the relation (source node).
        """
        assert excluded_relation != [] and valid_nodes != [] and test_nodes != []

        for name, relation in self:
            edges = relation.get_edges()
            print(f"Removing {name} relation for test artwork")
            edges = edges[~edges['source'].isin(test_nodes)]
            if name not in excluded_relation:
                print(f"Removing {name} relation for valid artwork")
                edges = edges[~edges['source'].isin(valid_nodes)]
            relation.add_edges(edges)

class NodeInstances():
    """A node type.

    It contains node type information, all its instances and attributes.
    Instances and attributes are stored in a pandas.DataFrame structure.   
    It has also the capability of saving itself on disk, according to the OGB convention.
    """
    def __init__(self, name):
        self.name = name
        self._instances = None
        self._attributes = {}

    def add_instances(self, instances):
        self._instances = instances
    
    def add_attributes(self, name, attributes):
        assert self._instances is not None
        assert len(self._instances) == len(attributes)

        self._attributes[name] = attributes

    def get_instances(self):
        return self._instances
    
    def get_all_attributes(self):
        return self._attributes
    
    def get_attributes(self, name):
        return self._attributes['name']
    
    @property
    def instances_number(self):
        return len(self._instances)
    
    def save_on_disk(self, root):
        file_path = os.path.join(root, self.name + "_entidx2name.csv")

        logging.info(f'Saving on the disk node mapping {self.name} in path {file_path}')

        self._instances.to_csv(os.path.join(file_path), index=True, header=False)

class SplitInstances(NodeInstances):
    """Node instances of the train, test and validation set.

    It is a subclass of NodeInstances, since each split contains node instances.
    It has also the capability of saving itself on disk, according to the OGB convention.
    """
    def __init__(self, name):
        assert name in ['train', 'test', 'valid']
        super().__init__(name)

    def save_on_disk(self, root):
        file_path = os.path.join(root, self.name + ".csv")

        logging.info(f'Saving on the disk node mapping {self.name} in path {file_path}')

        self._instances.to_csv(os.path.join(file_path), index=False, header=False)


class Nodes():
    """Node types of the graph and their relative information.
    It contains all the node types of the graph and their relative instances in a dictionary data structure.
    Each node is a object of the call raw_generation.NodeInstances.
    """
    def __init__(self):
        self._nodes = {}

    def add_node(self, name, node):
        self._nodes[name] = node

    def get_node(self, name):
        return self._nodes[name]

    def __iter__(self):
        return iter(self._nodes.items())
    
    def save_on_disk(self, root):
        logging.info('Saving on the disk all nodes')
        for _, node in self:
            node.save_on_disk(root)