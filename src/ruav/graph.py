"""Interaction graph utilities (Section 2.2) and the redundancy check of Assumption 1(iv)."""
from __future__ import annotations
from itertools import combinations
import numpy as np


def laplacian(A: np.ndarray) -> np.ndarray:
    """L = D - A for adjacency A[i, j] = 1 if follower i receives from follower j."""
    A = np.asarray(A, dtype=float)
    return np.diag(A.sum(axis=1)) - A


def pinned_laplacian(A: np.ndarray, g: np.ndarray) -> np.ndarray:
    """H = L + G, Eq. (3)."""
    return laplacian(A) + np.diag(np.asarray(g, dtype=float))


def reachable_from_leader(A: np.ndarray, g: np.ndarray) -> bool:
    """True if every follower is reachable from the leader (directed spanning tree rooted at the leader)."""
    A = np.asarray(A) > 0
    g = np.asarray(g) > 0
    n = len(g)
    seen = set(np.flatnonzero(g).tolist())
    frontier = list(seen)
    while frontier:
        j = frontier.pop()
        for i in range(n):            # i receives from j
            if A[i, j] and i not in seen:
                seen.add(i)
                frontier.append(i)
    return len(seen) == n


def in_links(A: np.ndarray, g: np.ndarray, i: int) -> list[int]:
    """Incoming links of follower i; the leader is encoded as -1."""
    links = [int(j) for j in np.flatnonzero(np.asarray(A)[i] > 0)]
    if np.asarray(g)[i] > 0:
        links.append(-1)
    return links


def leader_edge_connectivity(A: np.ndarray, g: np.ndarray) -> int:
    """Minimum, over followers i, of the number of edge-disjoint directed paths leader -> i.

    By Menger's theorem the augmented graph stays rooted at the leader after removing ANY
    k links if and only if this number is at least k + 1.
    """
    import networkx as nx
    A = np.asarray(A) > 0
    g = np.asarray(g) > 0
    n = len(g)
    D = nx.DiGraph()
    D.add_nodes_from(range(-1, n))            # -1 is the leader
    for i in range(n):
        if g[i]:
            D.add_edge(-1, i, capacity=1)
        for j in range(n):
            if A[i, j]:
                D.add_edge(j, i, capacity=1)  # information flows j -> i
    return min(int(nx.maximum_flow_value(D, -1, i)) for i in range(n))


def is_f_link_redundant(A: np.ndarray, g: np.ndarray, f: int) -> bool:
    """Assumption 1(iv): removing ANY f links of the augmented graph (leader links included)
    keeps a directed spanning tree rooted at the leader."""
    return leader_edge_connectivity(A, g) >= f + 1


def brute_force_f_link_redundant(A: np.ndarray, g: np.ndarray, f: int) -> bool:
    """Independent brute-force version of is_f_link_redundant, used in the tests."""
    A = np.asarray(A, dtype=int)
    g = np.asarray(g, dtype=int)
    n = len(g)
    links = [(j, i) for i in range(n) for j in in_links(A, g, i)]
    for removed in combinations(links, min(f, len(links))):
        A2, g2 = A.copy(), g.copy()
        for j, i in removed:
            if j == -1:
                g2[i] = 0
            else:
                A2[i, j] = 0
        if not reachable_from_leader(A2, g2):
            return False
    return True
