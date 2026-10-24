import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ruav import dynamics, graph, attacks  # noqa: E402

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "hybrid_attack.yaml")


class TestDynamics(unittest.TestCase):
    def test_constant_acceleration_exact(self):
        p, v = np.zeros((1, 3)), np.zeros((1, 3))
        u = np.array([[1.0, 0.0, 0.0]])
        z = np.zeros((1, 3))
        for _ in range(100):
            p, v = dynamics.step(p, v, u, z, z, 0.01)
        self.assertAlmostEqual(p[0, 0], 0.5, places=9)   # 0.5 * a * t^2, t = 1 s
        self.assertAlmostEqual(v[0, 0], 1.0, places=9)

    def test_saturation(self):
        u = dynamics.saturate(np.array([[3.0, 4.0, 0.0]]), 2.0)
        self.assertAlmostEqual(np.linalg.norm(u), 2.0)


class TestGraph(unittest.TestCase):
    def test_pinned_laplacian_eigs_positive(self):
        A = np.array([[0, 1], [1, 0]]); g = np.array([1, 0])
        H = graph.pinned_laplacian(A, g)
        self.assertTrue(np.all(np.linalg.eigvals(H).real > 0))

    def test_chain_not_redundant(self):
        A = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]); g = np.array([1, 0, 0])
        self.assertTrue(graph.reachable_from_leader(A, g))
        self.assertFalse(graph.is_f_link_redundant(A, g, f=1))

    def test_flow_matches_brute_force(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            n = int(rng.integers(3, 7))
            A = (rng.random((n, n)) < 0.4).astype(int); np.fill_diagonal(A, 0)
            g = (rng.random(n) < 0.4).astype(int)
            for f in (0, 1, 2):
                self.assertEqual(graph.is_f_link_redundant(A, g, f),
                                 graph.brute_force_f_link_redundant(A, g, f))

    def test_benchmark_graph_is_1_redundant(self):
        import yaml
        cfg = yaml.safe_load(open(CFG))
        A = np.array(cfg["graph"]["adjacency"]); g = np.array(cfg["graph"]["pinning"])
        self.assertTrue(graph.is_f_link_redundant(A, g, f=1))


class TestAttacks(unittest.TestCase):
    def test_schedule(self):
        s = attacks.AttackSchedule.from_yaml(CFG)
        self.assertTrue(s.is_time_staggered_multichannel())
        self.assertEqual(s.n_switches(0, 120), 6)
        self.assertAlmostEqual(s.attacked_time(0, 120, dt=0.01), 45.0, delta=0.05)
        self.assertEqual(len(s.active_events(37.0)), 2)   # sensor + comm overlap


if __name__ == "__main__":
    unittest.main()
