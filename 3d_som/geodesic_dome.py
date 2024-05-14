import numpy as np
import math
import numba
from numba import njit
from numba.typed import Dict
from typing import Union


class GeodesicDome:
    """Class wrapper to create and interact with a Geodesic Dome"""

    def __init__(self, freq=0) -> None:
        """Creates a given geodesic dome with a given frequency.

        Args:
            freq (int, optional): The frequency of the geodesic dome. Defaults to 0.
        """
        self.vertices, self.triangles, self.adj_list = create_geodesic_dome(
            freq)
        pass

    def tessellate(self, freq=1) -> None:
        """Tessellates the geodesic dome a given number of times (tessellates once if no arguments provided)

        Args:
            freq (int, optional): The number of times to tessellate. Defaults to 1.
        """
        for i in range(freq):
            self.vertices, self.triangles, self.adj_list = tessellate_geodesic_dome(
                self.vertices, self.triangles
            )

        return

    def find_neighbours(self, index: np.int64, depth=1) -> np.ndarray:
        """Finds the neighbours of a given vertex on the geodesic dome to a certain depth (defaults to 1 if not provided)

        Args:
            index (np.int64): The index of the vertex to search from
            depth (int, optional): The depth of neighbours to return. Defaults to 1.

        Returns:
            np.ndarray: An array containing the indices of all the vertex's neighbours
        """
        return find_neighbours(self.vertices, self.adj_list, index, depth)

    def get_vertices(self) -> np.ndarray:
        """Getter function for vertices

        Returns:
            np.ndarray: the vertices of the geodesic dome
        """
        return self.vertices

    def get_triangles(self) -> np.ndarray:
        """Getter function for triangles

        Returns:
            np.ndarray: the triangles of the geodesic dome
        """
        return self.triangles


SCALE = 1


@njit
def is_zero(coord, message="") -> None:
    """Debug method, checks if a coord is at origin

    Args:
        coord (np array): coordinate to check
        message (str, optional): descriptive message when triggered. Defaults to "".
    """
    if (coord == np.array([0, 0, 0])).all():
        print("Something is zero that shouldn't be: " + message)


@njit
def normalise_length(coords: np.ndarray) -> np.ndarray:
    """Normalises the distance from origin of a coord. Multiplies by the
    frequency the icosphere to avoid floating point precision errors

    Args:
        coords (np.ndarray): coordinate to normalise

    Returns:
        np.ndarray: normalised coordinate
    """
    length = math.sqrt(
        math.pow(coords[0], 2) + math.pow(coords[1], 2) +
        math.pow(coords[2], 2)
    )

    is_zero(coords, "normalise")

    return np.array(
        [
            (coords[0] / length) * SCALE,
            (coords[1] / length) * SCALE,
            (coords[2] / length) * SCALE,
        ]
    )


@njit
def get_middle_coords(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Gets the midpoint between two coords

    Args:
        v1 (np.ndarray): coord 1
        v2 (np.ndarray): coord 2

    Returns:
        np.ndarray: the midpoint (not normalised)
    """
    ret = np.array(
        [(v2[0] + v1[0]) / 2, (v2[1] + v1[1]) / 2, (v2[2] + v1[2]) / 2],
        dtype=np.float64,
    )
    return ret


@njit
def add_middle_to_vertices(mid: np.ndarray, vertices: np.ndarray, v_index: Dict) -> int:
    """Adds a given midpoint to a list of new vertices

    Args:
        mid (np.ndarray): the midpoint to add
        vertices (np.ndarray): an array of new vertices, to be concatenated with
        existing vertices
        v_index (Dict): dictionary containing the indexes of existing midpoints,
        to prevent duplicates

    Returns:
        int: the index of the midpoint that was added to vertices
    """
    # Creating (hopefully) unique key for each coordinate
    mid_sum = mid[0] * 3 + mid[1] * 2 + mid[2]
    # Adding key to dictionary of coords with index as value
    if mid_sum not in v_index:
        v_index[mid_sum] = len(v_index)
    index = v_index[mid_sum]
    # Add new midpoint to new vertices array
    if (vertices[index] == np.array([0, 0, 0])).all():
        vertices[index] = mid

    return index


@njit
def normalise_all(new_vertices: np.ndarray) -> None:
    """Normalises all the vertices in an array

    Args:
        new_vertices (np.ndarray): the array of vertices
    """
    for i in range(len(new_vertices)):
        new_vertices[i] = normalise_length(new_vertices[i])


@njit
def adj_insert(adj: np.ndarray, root: np.int64, neighbour: np.int64) -> None:
    """Function to insert a point into adjacency list of root vertex

    Args:
        adj (np.ndarray): array of arrays, representing adjacency list
        root (np.int64): index of root vertex
        neighbour (np.int64): index of neighbour vertex to add
    """
    root_list = adj[root]
    for i in range(6):
        if root_list[i] == neighbour:
            break
        if root_list[i] == -1:
            root_list[i] = neighbour
            break


@njit
def create_adj_list(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Function to create adjacency list representation of vertices

    Args:
        vertices (np.ndarray): numpy array of vertices
        triangles (np.ndarray): numpy array of vertices

    Returns:
        np.ndarray: array of arrays representing adjacency list
    """
    adj = np.full((len(vertices), 6), -1, dtype=np.int64)

    for t in triangles:
        adj_insert(adj, t[0], t[1])
        adj_insert(adj, t[0], t[2])
        adj_insert(adj, t[1], t[0])
        adj_insert(adj, t[1], t[2])
        adj_insert(adj, t[2], t[0])
        adj_insert(adj, t[2], t[1])

    return adj


@njit
def generate_new_points(
    vertices: np.ndarray, triangles: np.ndarray, t: int
) -> Union[np.ndarray, np.ndarray]:
    """Tesselates the entire icosphere once. Returns an array containing the
    new vertices, to be concatenated with existing vertices, and a set of new
    triangles, to replace the old triangles

    Args:
        vertices (np.ndarray): the current vertices in the icosphere
        triangles (np.ndarray): the current triangles in the icosphere
        t (int): triangulation factor, used to calculate sizes of new vertices and triangles arrays

    Returns:
        np.ndarray: array of new vertices
        np.ndarray: array of new triangles
    """
    # create new array for new triangles
    new_triangles = np.zeros((len(triangles) * 4, 3), dtype=numba.int64)
    n_old_vertices = 10 * int(t / 4) + 2
    n_new_vertices = 10 * int(t - t / 4)
    # create new array for new vertices
    new_vertices = np.zeros((n_new_vertices, 3), dtype=numba.float64)

    i = 0
    v_index = Dict.empty(
        key_type=np.float64,
        value_type=np.int64,
    )
    for tri in triangles:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]

        is_zero(v0, "vertex")
        is_zero(v1, "vertex")
        is_zero(v2, "vertex")

        # Get midpoints for each edge of the triangle
        mid01 = get_middle_coords(v0, v1)
        mid12 = get_middle_coords(v1, v2)
        mid02 = get_middle_coords(v2, v0)

        # Get indexes of the new midpoints with respect to current vertices
        index01 = add_middle_to_vertices(
            mid01, new_vertices, v_index) + n_old_vertices
        index12 = add_middle_to_vertices(
            mid12, new_vertices, v_index) + n_old_vertices
        index02 = add_middle_to_vertices(
            mid02, new_vertices, v_index) + n_old_vertices

        # Create new triangles
        new_triangles[i] = [tri[0], index01, index02]
        new_triangles[i + 1] = [tri[1], index12, index01]
        new_triangles[i + 2] = [tri[2], index02, index12]
        new_triangles[i + 3] = [index01, index12, index02]

        i += 4
    return new_vertices, new_triangles


@njit
def calc_dist(points) -> list:
    """Calculates the distance of each point in the Dome from the origin

    Args:
        points (list): list of points

    Returns:
        list: List of distances of each point in the Dome
    """

    distances = []
    for p in points:
        dist = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
        distances.append(round(dist, 2))

    return distances


def create_js_json(vertices: np.ndarray, triangles: np.ndarray) -> None:
    """Generates a json file for 3js visualisation from an icosphere

    Args:
        vertices (np.ndarray): the vertices array
        triangles (np.ndarray): the triangles array
    """
    f = open("3js/icojson.js", "w")
    f.write('export default {\n  "vertices": [\n')
    for vertex in vertices:
        for coord in vertex:
            f.write(f"{coord},")
    f.write('], "indices": [\n')
    for triangle in triangles:
        for coord in triangle:
            f.write(f"{coord},")
    f.write('], "radius": 1,')
    f.write('"details": 0}')
    f.close()


@njit
def find_neighbours(
    vertices: np.ndarray, adj_list: np.ndarray, index: np.int64, depth=1
) -> np.ndarray:
    """Function to find nearest neighbours to a specific point, up to a
    specified depth

        Args:
            vertices (np.ndarray): numpy array of vertices in the Dome
            adj (np.ndarray): adjacency list of the vertices
            index (np.int64): index of the root vertex
            depth (np.int64, optional): search depth. Defaults to 1.

        Returns:
            np.ndarray: Array of neighbours found, may include -1 representing empty entries
    """
    size = 0
    for i in range(depth):
        size += (i + 1) * 6

    if size > len(vertices) - 1:
        size = len(vertices) - 1

    curr_depth = 1

    neighbours = np.full(size, -1, dtype=np.int64)
    num_neighbours = 0
    queue = np.full(1, index, dtype=np.int64)
    visited = np.full(len(vertices), False, dtype=np.bool_)

    q_end = 1

    while curr_depth <= depth:
        temp = np.full(curr_depth * 6, -1, dtype=np.int64)
        temp_ptr = 0
        q_front = 0

        while q_front < q_end:
            v_index = queue[q_front]
            for neighbour in adj_list[v_index]:
                if neighbour != -1 and visited[neighbour] == False:
                    neighbours[num_neighbours] = neighbour
                    temp[temp_ptr] = neighbour

                    num_neighbours += 1
                    temp_ptr += 1
                    visited[neighbour] = True
            visited[v_index] = True
            q_front += 1

        queue = temp
        q_end = temp_ptr
        curr_depth += 1
        if temp_ptr == 0:
            break

    return neighbours


@njit
def tessellate_geodesic_dome(
    vertices, triangles
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """Tesselates the entire icosphere once. Returns an array containing the
    new vertices, to be concatenated with existing vertices, and a set of new
    triangles, to replace the old triangles

    Args:
        vertices (np.ndarray): the current vertices in the icosphere
        triangles (np.ndarray): the current triangles in the icosphere

    Returns:
        np.ndarray: array of new vertices
        np.ndarray: array of new triangles
        np.ndarray: array of arrays representing adjacency list
    """
    t = ((len(vertices) - 2) / 10) * 4

    # create new array for new triangles
    new_triangles = np.zeros((len(triangles) * 4, 3), dtype=np.int64)
    n_old_vertices = len(vertices)
    n_new_vertices = 10 * int(t - t / 4)
    # create new array for new vertices
    new_vertices = np.zeros((n_new_vertices, 3), dtype=np.float64)

    i = 0
    v_index = Dict.empty(
        key_type=np.float64,
        value_type=np.int64,
    )

    for tri in triangles:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]

        # Get midpoints for each edge of the triangle
        mid01 = get_middle_coords(v0, v1)
        mid12 = get_middle_coords(v1, v2)
        mid02 = get_middle_coords(v2, v0)

        # Get indexes of the new midpoints with respect to current vertices
        index01 = (
            add_middle_to_vertices(
                mid01, new_vertices, v_index) + n_old_vertices
        )
        index12 = (
            add_middle_to_vertices(
                mid12, new_vertices, v_index) + n_old_vertices
        )
        index02 = (
            add_middle_to_vertices(
                mid02, new_vertices, v_index) + n_old_vertices
        )

        # Create new triangles
        new_triangles[i] = [tri[0], index01, index02]
        new_triangles[i + 1] = [tri[1], index12, index01]
        new_triangles[i + 2] = [tri[2], index02, index12]
        new_triangles[i + 3] = [index01, index12, index02]

        i += 4
    normalise_all(new_vertices)
    # Keep track of all previous vertices
    old_vertices = vertices
    # Create array to concatenate old vertices with new midpoints
    vertices = np.zeros(
        (len(old_vertices) + len(new_vertices), 3), dtype=np.float64)

    i = 0
    # Add old vertices
    for v in old_vertices:
        vertices[i] = v
        i += 1
    # Add new midpoints
    for v in new_vertices:
        vertices[i] = v
        i += 1

    new_adj_list = create_adj_list(vertices, new_triangles)
    return vertices, new_triangles, new_adj_list


@njit
def create_geodesic_dome(freq=0) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """Creates an geodesic dome of a given frequency

    Args:
        freq (int, optional): the frequency of the dome. Defaults to 0.

    Returns:
        Union[np.ndarray, np.ndarray]: the array of vertices, the array of
        triangles

        Vertices = [[x,y,z], ... , [x,y,z]]
        Triangles = [[v1, v2, v3], ...] where vx is the index of a vertex in the vertices array

        Adjacency list = [[v1, ..., v5, v6?], ...] where vx is the index of a vertex. v6 may not exist for some vertices
    """
    # Set normalised scaling
    if freq != 0:
        SCALE = freq

    g_ratio = (1 + math.sqrt(5)) / 2

    # creating initial icosahedron vertices
    icosa_vertices = np.array(
        [
            (-1, g_ratio, 0),
            (1, g_ratio, 0),
            (-1, -(g_ratio), 0),
            (1, -(g_ratio), 0),
            (0, -1, g_ratio),
            (0, 1, g_ratio),
            (0, -1, -(g_ratio)),
            (0, 1, -(g_ratio)),
            (g_ratio, 0, -1),
            (g_ratio, 0, 1),
            (-(g_ratio), 0, -1),
            (-(g_ratio), 0, 1),
        ],
        dtype=np.float64,
    )
    # creating initial icosahedron edges
    triangles = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int64,
    )

    # Array for normalised vertices
    vertices = np.zeros((len(icosa_vertices), 3), dtype=np.float64)

    # Normalise all icosahedron vertices
    for i in range(len(icosa_vertices)):
        vertices[i] = normalise_length(icosa_vertices[i])
    # Tessellate icosahedron
    adj_list = create_adj_list(vertices, triangles)
    for i in range(freq):
        vertices, triangles, adj_list = tessellate_geodesic_dome(
            vertices, triangles)
    return vertices, triangles, adj_list
