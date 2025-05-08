from scipy.sparse.csgraph import dijkstra
from copy import deepcopy
import heapq
import numpy as np
from scipy.sparse import csr_matrix


def max_bottleneck_path(connectivity_matrix: csr_matrix,
                        start_node: int,
                        end_node: int) -> (list, float):
    """
    Finds the path between start_node and target that maximizes the minimum connectivity (bottleneck).

    Parameters
    ----------
    connectivity_matrix : csr_matrix
        A scipy.sparse.csr_matrix representing the connectivity graph.
    start_node : int
        The starting node index.
    end_node : int
        The destination node index.

    Returns
    -------
    list
        The path as a list of nodes.
    float
        Maximum bottleneck value
    """
    num_nodes = connectivity_matrix.shape[0]
    max_bottleneck = np.full(num_nodes, -np.inf)  # Initialize bottleneck values
    max_bottleneck[start_node] = np.inf  # Start with infinite connectivity
    parent = {start_node: None}  # Dictionary to reconstruct path

    # Max-heap priority queue (negative values to simulate max-heap with heapq)
    pq = [(-np.inf, start_node)]

    while pq:
        bottleneck, node = heapq.heappop(pq)
        bottleneck = -bottleneck  # Convert back to positive value

        if node == end_node:
            break  # Early exit if end_node reached

        for neighbor in connectivity_matrix.indices[connectivity_matrix.indptr[node]:connectivity_matrix.indptr[node + 1]]:
            weight = connectivity_matrix[node, neighbor]  # Connectivity value
            new_bottleneck = min(bottleneck, weight)

            if new_bottleneck > max_bottleneck[neighbor]:
                max_bottleneck[neighbor] = new_bottleneck
                parent[neighbor] = node
                heapq.heappush(pq, (-new_bottleneck, neighbor))

    if max_bottleneck[end_node] == -np.inf:
        return None, []  # No path found

    # Reconstruct path
    path = []
    node = end_node
    while node is not None:
        path.append(node)
        node = parent.get(node, None)
    path.reverse()

    return path, max_bottleneck[end_node]


def find_min_cost_path(connectivity_matrix: csr_matrix,
                       start_node: int,
                       end_node: int) -> (list, float):
    """
    Identify the path in the graph that has the minimum cost. Assumes that the input is connectivity; the cost is set
    as 1-connectivity_matrix.
    Parameters
    ----------
    connectivity_matrix : csr_matrix
        A scipy.sparse.csr_matrix representing the connectivity graph.
    start_node : int
        The starting node index.
    end_node : int
        The destination node index.

    Returns
    -------

    """
    # Compute the shortest path distance between all pairs of nodes in the graph
    connectivity_matrix = deepcopy(connectivity_matrix)  # We need to modify the values while preserving the original
    connectivity_matrix[connectivity_matrix.nonzero()] = 1-connectivity_matrix[connectivity_matrix.nonzero()]

    distances, predecessors, _ = dijkstra(csgraph=connectivity_matrix,
                                          directed=False,
                                          indices=start_node,
                                          return_predecessors=True,
                                          min_only=True)
    path = []

    # Start with the end_node and loop until start_node is reached
    current_node = end_node
    while current_node != start_node and current_node != -9999:
        # Prepend the current_node to the path
        path.insert(0, current_node)

        # Move towards the predecessor of the current_node
        current_node = predecessors[current_node]

    # Check if we reached the start_node
    if current_node != -9999:
        path.insert(0, start_node)
    else:
        return None, -np.inf
    # Return the path and its total cost
    return path, distances[end_node]
