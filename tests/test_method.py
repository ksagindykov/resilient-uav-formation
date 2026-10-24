import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ruav import cbf, detector, dynamics  # noqa: E402


class TestCBF(unittest.TestCase):
    def _run(self, use_filter, attack):
        """Two followers flying head-on at 2 m/s; follower 0 suffers an actuation attack."""
        prm = cbf.CBFParams(d_safe=1.0, a_u_bar=0.8, d_bar=0.0, u_box=6.0)
        p = np.array([[-5.0, 0.05, 0.0], [5.0, -0.05, 0.0]])
        v = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        dt, dmin = 0.01, np.inf
        for k in range(600):
            u = np.zeros((2, 3))
            for i, j in ((0, 1), (1, 0)):
                u_nom = np.zeros(3)
                if use_filter:
                    rel = {j: (p[j] - p[i], v[j] - v[i])}
                    u[i], _ = cbf.safety_filter(u_nom, rel, prm)
            a_u = np.zeros((2, 3))
            if attack:
                a_u[0] = 0.8 * (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])  # pushes towards j
            p, v = dynamics.step(p, v, u, a_u, np.zeros((2, 3)), dt)
            dmin = min(dmin, np.linalg.norm(p[0] - p[1]))
        return dmin

    def test_without_filter_collides(self):
        self.assertLess(self._run(False, True), 1.0)

    def test_filter_keeps_distance_under_bounded_actuation_attack(self):
        self.assertGreaterEqual(self._run(True, True), 1.0 - 1e-3)

    def test_qp_inactive_when_safe(self):
        prm = cbf.CBFParams()
        u0 = np.array([0.3, -0.2, 0.1])
        u, info = cbf.safety_filter(u0, {1: (np.array([5.0, 0, 0]), np.zeros(3))}, prm)
        np.testing.assert_allclose(u, u0, atol=1e-9)
        self.assertEqual(info["violation"], 0.0)


class TestDetector(unittest.TestCase):
    def _feed(self, sensor_bias, link_bias):
        prm = detector.DetectorParams(n_on=3, n_off=10)
        det = detector.Detector(prm, neighbours=[1, 2, 3], dt=0.02)
        p_true = {0: np.zeros(3), 1: np.array([2.0, 0, 0]), 2: np.array([0, 2.0, 0]),
                  3: np.array([-2.0, 0, 0])}
        out = None
        for _ in range(10):
            y = np.concatenate([p_true[0] + sensor_bias, np.zeros(3)])
            msgs = {j: np.concatenate([p_true[j], np.zeros(3)]) for j in (1, 2, 3)}
            msgs[2] = msgs[2] + np.concatenate([link_bias, np.zeros(3)])
            rel = {j: (p_true[j] - p_true[0], np.zeros(3)) for j in (1, 2, 3)}
            out = det.update(y, np.zeros(3), msgs, rel)
        return out

    def test_clean(self):
        o = self._feed(np.zeros(3), np.zeros(3))
        self.assertFalse(o.sensor_alarm or any(o.link_alarm.values()))

    def test_link_fdi_isolated(self):
        o = self._feed(np.zeros(3), np.array([1.0, 0, 0]))
        self.assertFalse(o.sensor_alarm)
        self.assertTrue(o.link_alarm[2]); self.assertFalse(o.link_alarm[1])

    def test_own_sensor_attack(self):
        o = self._feed(np.array([1.0, -0.7, 0.3]), np.zeros(3))
        self.assertTrue(o.sensor_alarm)
        self.assertFalse(any(o.link_alarm.values()))


if __name__ == "__main__":
    unittest.main()
