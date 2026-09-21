"""
RoadNetwork Module

This module provides the RoadNetwork class for managing, processing, and converting road network data,
including downloading, loading, and generating matrices required for MST-OATD.
"""

import logging
import pickle
from os.path import exists

import numpy as np
import osmnx as ox
from geopandas import GeoDataFrame, gpd
from networkx import MultiDiGraph
from scipy import sparse

logger = logging.getLogger(__name__)


class RoadNetwork:
    """
    Class for managing and processing road network data for a given location.

    Attributes:
        location (str): The name of the location.
        graph_file (str): Path to the graphml file.
        nodes_file (str): Path to the nodes shapefile.
        edges_file (str): Path to the edges shapefile.
        mst_A_matrix_file (str): Path to the adjacency matrix file.
        mst_D_matrix_file (str): Path to the normalized degree matrix file.
    """

    def __init__(self, location: str):
        """
        Initialize the RoadNetwork object with file paths for the given location.

        Args:
            location (str): The name of the location.
        """
        self.location = location
        self.graph_file = f"data/{location}/raw/graph.graphml"
        self.nodes_file = f"data/{location}/raw/nodes.shp"
        self.edges_file = f"data/{location}/raw/edges.shp"
        self.mst_A_matrix_file = f"data/{location}/raw/adj.npz"
        self.mst_D_matrix_file = f"data/{location}/raw/d_norm.npz"
        self.cause_tad_required_adj_dict = f"data/{location}/raw/adj_dict.pkl"

    def __call__(self) -> int:
        """
        Prepare the road network data and required matrices. If files exist, loads and processes them;
        otherwise, downloads and converts the road network.

        Returns:
            int: The number of edges in the road network.
        """

        if self.is_edges_nodes_exists():
            logger.info("edges and nodes .csv file exist")
            edges_gdf = gpd.read_file(self.edges_file)
            self.generate_mst_required_matrix(edges_gdf)
            self.generate_cause_tad_required_adj_dict(edges_gdf)
            num_edges = len(edges_gdf)
            return num_edges

        if self.is_road_network_exist():
            g = self.load_road_network()
        else:
            g = self.download_road_network()

        num_edges = self.convert_road_network(g)
        edges_gdf = gpd.read_file(self.edges_file)
        self.generate_mst_required_matrix(edges_gdf)
        self.generate_cause_tad_required_adj_dict(edges_gdf)
        return num_edges
    
    def get_num_edges(self) -> int:
        """
        Get the number of edges in the road network.

        Returns:
            int: The number of edges.
        """

        edges_gdf = gpd.read_file(self.edges_file)
        return len(edges_gdf)

    def get_cause_tad_required_adj_dict(self, edges_gdf: GeoDataFrame) -> dict:
        """Generate the adjacency dictionary required for CausalTAD.
        Args:
            edges_gdf (GeoDataFrame): GeoDataFrame containing edge data.
        Returns:
            dict: Adjacency dictionary where keys are edge IDs and values are lists of adjacent edge IDs.
        """
        logger.info("Get CausalTAD required adjacency dictionary")
        
        # Pre-compute edge mapping for each node
        node_to_fids = {}
        for fid, u, v in zip(edges_gdf["fid"], edges_gdf["u"], edges_gdf["v"]):
            node_to_fids.setdefault(u, []).append(fid)
            node_to_fids.setdefault(v, []).append(fid)
            
        adj_dict = {}
        for i, (fid, u, v) in enumerate(zip(edges_gdf["fid"], edges_gdf["u"], edges_gdf["v"])):
            if fid % 1000 == 0:
                logger.info("Generating adjacency dictionary: %d/%d", i, len(edges_gdf))
            
            # Combine all edges connected to nodes u and v
            adj_set = set(node_to_fids[u] + node_to_fids[v])
            adj_set.discard(fid)  # Remove self
            
            adj_dict[fid] = list(adj_set)
            
        return adj_dict

    def generate_cause_tad_required_adj_dict(self, edges_gdf: GeoDataFrame):
        """
        Generate and save the adjacency dictionary required for CausalTAD if it does not exist.

        Args:
            edges_gdf (GeoDataFrame): GeoDataFrame containing edge data.
        """

        if not self.is_cause_tad_required_adj_dict_exists():
            adj_dict = self.get_cause_tad_required_adj_dict(edges_gdf)
            with open(self.cause_tad_required_adj_dict, "wb") as f:
                pickle.dump(adj_dict, f)
                logger.info(
                    "CausalTAD required adjacency dictionary saved to %s",
                    self.cause_tad_required_adj_dict,
                )

    def is_cause_tad_required_adj_dict_exists(self) -> bool:
        """
        Check if the CausalTAD required adjacency dictionary exists.

        Returns:
            bool: True if the adjacency dictionary exists, False otherwise.
        """

        result = exists(self.cause_tad_required_adj_dict)
        if result:
            logger.info("CausalTAD required adjacency dictionary exists")
        else:
            logger.info("CausalTAD required adjacency dictionary does not exist")
        return result

    def generate_mst_required_matrix(self, edges_gdf: GeoDataFrame):
        """
        Generate and save the adjacency and normalized degree matrices required for MST-OATD if they do not exist.

        Args:
            edges_gdf (GeoDataFrame): GeoDataFrame containing edge data.
        """

        logger.info("Get MST-OATD required matrix A and D")
        if not self.is_mst_required_matrix_exists():
            a, d = self.get_a_d_matrix(edges_gdf)
            sparse.save_npz(self.mst_A_matrix_file, a)
            sparse.save_npz(self.mst_D_matrix_file, d)

    def is_mst_required_matrix_exists(self):
        """
        Check if the MST-OATD required matrices exist.

        Returns:
            bool: True if both matrices exist, False otherwise.
        """

        result = exists(self.mst_A_matrix_file) and exists(self.mst_D_matrix_file)
        if result:
            logger.info("Matrix D or A .npz files exist")
        else:
            logger.info("Matrix D or A .npz files not exist")
        return result

    def is_edges_nodes_exists(self) -> bool:
        """
        Check if the edges and nodes shapefiles exist.

        Returns:
            bool: True if both files exist, False otherwise.
        """

        result = exists(self.nodes_file) and exists(self.edges_file)
        if result:
            logger.info("Edges and nodes .shp files exist")
        else:
            logger.info("Edges and nodes .shp files not exist")
        return result

    def is_road_network_exist(self) -> bool:
        """
        Check if the road network graphml file exists.

        Returns:
            bool: True if the file exists, False otherwise.
        """

        result = exists(self.graph_file)
        if result:
            logger.info("Road network exist in %s", self.graph_file)
        else:
            logger.info("Road network not exist")
        return result

    def load_road_network(self) -> MultiDiGraph:
        """
        Load the road network from the graphml file.

        Returns:
            MultiDiGraph: The loaded road network graph.
        """

        logger.info("Load road network from %s", self.graph_file)
        g = ox.load_graphml(self.graph_file)
        return g

    def download_road_network(self) -> MultiDiGraph:
        """
        Download the road network for the specified location and save it as a graphml file.

        Returns:
            MultiDiGraph: The downloaded road network graph.
        """
        if self.location == "xian":
            location = "xi'an"
        else:
            location = self.location
        logger.info("Downloading road network for %s", location)
        g = ox.graph_from_place(location, network_type="drive")
        logger.info("Save road network to %s", self.graph_file)
        ox.save_graphml(g, self.graph_file)
        return g

    def convert_road_network(self, g: MultiDiGraph) -> int:
        """
        Convert the road network graph to GeoDataFrames, assign unique IDs, and save as shapefiles.

        Args:
            g (MultiDiGraph): The road network graph.
        Returns:
            int: The number of edges in the network.
        """

        logger.info("Convert road network to GeoDataFrame")
        gdf_node, gdf_edge = ox.graph_to_gdfs(g)
        gdf_node = ox.io._stringify_nonnumeric_cols(gdf_node)
        gdf_edge = ox.io._stringify_nonnumeric_cols(gdf_edge)

        logger.info("Assign unique id to each edge")
        gdf_edge["fid"] = np.arange(0, gdf_edge.shape[0], dtype="int")
        gdf_node["fid"] = np.arange(0, gdf_node.shape[0], dtype="int")

        logger.info("Network nodes count: %d", len(gdf_node))
        logger.info("Network edges count: %d", len(gdf_edge))

        logger.info("Save nodes to shp file in %s", self.nodes_file)
        gdf_node.to_file(self.nodes_file, encoding="utf-8")
        logger.info("Save edges to shp file in %s", self.edges_file)
        gdf_edge.to_file(self.edges_file, encoding="utf-8")

        return len(gdf_edge)

    def get_a_d_matrix(self, edge_gdf: GeoDataFrame):
        """
        Generate the adjacency and normalized degree matrices from the edge GeoDataFrame.

        Args:
            edge_gdf (GeoDataFrame): GeoDataFrame containing edge data.
        Returns:
            tuple: (adjacency matrix, normalized degree matrix)
        """

        a = self.get_adjacency_matrix(edge_gdf)
        i = sparse.identity(len(edge_gdf))
        adj = a + i
        degree_list = np.array(adj.sum(axis=1)).flatten()
        inv_sqrt_degree = 1.0 / (np.sqrt(degree_list) + 1e-10)
        inv_sqrt_degree[np.isinf(inv_sqrt_degree)] = 0.0
        d = sparse.diags(inv_sqrt_degree, format="csr")
        return a + i, d

    def get_adjacency_matrix(self, edge_gdf: GeoDataFrame) -> sparse.coo_matrix:
        """
        Generate the adjacency matrix for the road network from the edge GeoDataFrame.

        Args:
            edge_gdf (GeoDataFrame): GeoDataFrame containing edge data.
        Returns:
            sparse.coo_matrix: The adjacency matrix in COO format.
        """

        node_to_fids = {}
        for fid, u, v in zip(edge_gdf["fid"], edge_gdf["u"], edge_gdf["v"]):
            node_to_fids.setdefault(u, []).append(fid)
            node_to_fids.setdefault(v, []).append(fid)

        row_indices = []
        col_indices = []
        values = []
        for i, (fid, u, v) in enumerate(zip(edge_gdf["fid"], edge_gdf["u"], edge_gdf["v"])):
            if fid % 1000 == 0:
                logger.info("Generating adjacent matrix: %d/%d", i, len(edge_gdf))
            
            adj_set = set(node_to_fids[u] + node_to_fids[v])
            adj_set.discard(fid)
            
            for temp_fid in adj_set:
                row_indices.append(fid)
                col_indices.append(temp_fid)
                values.append(1)

        return sparse.coo_matrix(
            arg1=(values, (row_indices, col_indices)),
            shape=(len(edge_gdf), len(edge_gdf)),
            dtype=int,
        )
