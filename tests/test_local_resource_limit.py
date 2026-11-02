import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_local_model_once import monitor_peak_rss


class LocalInferenceMemoryLimitTests(unittest.TestCase):
    def test_excess_rss_records_failure_and_terminates_process_group_callback(self):
        values = iter((300, 513))
        stop = threading.Event()
        exceeded = threading.Event()
        peak = [0]
        terminated = []

        def fake_terminate():
            terminated.append(True)
            stop.set()

        monitor_peak_rss(
            pid=123,
            stop_event=stop,
            exceeded_event=exceeded,
            peak_rss_kib=peak,
            ceiling_bytes=512 * 1024,
            read_rss=lambda _pid: next(values),
            terminate=fake_terminate,
            poll_interval=0,
        )
        self.assertEqual(peak[0], 513)
        self.assertTrue(exceeded.is_set())
        self.assertEqual(terminated, [True])


if __name__ == "__main__":
    unittest.main()
