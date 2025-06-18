from decentralizepy.graphs.Regular import Regular

random_seed = 90

graph = Regular(
            16,
            4,
            seed=random_seed,
        )
graph.read_graph_from_file("./c1_a1_k3_sta-2025-06-16T13:10/machine0/graph_-1.txt")

graph.write_graph_to_file("./graph_test.txt", type="edges")

def get_distance(graph, node1, node2):
    """
    Returns the distance between two nodes in the graph.
    """
    
    if node1 == node2:
        return 0
    visited = set()
    queue = [(node1, 0)]  # (current_node, current_distance)
    
    while queue:
        current_node, current_distance = queue.pop(0)
        if current_node == node2:
            return current_distance
        
        if current_node not in visited:
            visited.add(current_node)
            for neighbour in graph.neighbors(current_node):
                if neighbour not in visited:
                    queue.append((neighbour, current_distance + 1))
    
    
print(get_distance(graph, 0, 2))

special_nodes = [2]

def get_distance_to_special_nodes(graph, node):
    """
    Returns the distance from a node to the nearest special node.
    """
    distances = [get_distance(graph, node, special_node) for special_node in special_nodes]
    return min(distances)

print(get_distance_to_special_nodes(graph, 2))

for node in range(graph.n_procs):
    print(f"Node {node} distance to special nodes: {get_distance_to_special_nodes(graph, node)}")