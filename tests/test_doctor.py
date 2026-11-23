"""harness/doctor 与 bench/hw/dcu 解析逻辑的单测（不依赖 GPU）。

用假 hy-smi 输出驱动解析器：解析错的表现是"读到了错误的 MHz"，
而错误的 MHz 会静默进入 trace，比抛异常危险得多。
"""
from __future__ import annotations

import unittest
from unittest import mock

from bench.hw import dcu
import harness.doctor as doctor

SAMPLE_G = """
================================= System Management Interface ==================================
================================================================================================
HCU[0]		: sclk clock level: 5 (1200Mhz)
HCU[1]		: sclk clock level: 8 (1350Mhz)
================================================================================================
"""

SAMPLE_U = """
HCU[0]		: HCU use (%): 3.0
HCU[1]		: HCU use (%): 97.5
"""

SAMPLE_P = """
HCU[0]		: Performance Level: auto
HCU[1]		: Performance Level: manual
"""


class TestSclkParsing(unittest.TestCase):
    def test_parses_level_and_mhz(self):
        with mock.patch.object(dcu, "_run", side_effect=lambda a, timeout=15.0: SAMPLE_G):
            d = dcu.current_sclk_detail()
        self.assertEqual(d["levels"], [5, 8])
        self.assertEqual(d["mhz"], [1200, 1350])

    def test_min_not_max(self):
        """一张卡降频就整机按降频算——拿快卡的读数当机器状态会测成工作档。"""
        with mock.patch.object(dcu, "_run", side_effect=lambda a, timeout=15.0: SAMPLE_G):
            self.assertEqual(dcu.current_sclk(), 1200)
            self.assertEqual(dcu.current_sclk_detail()["min_level"], 5)

    def test_garbage_returns_none(self):
        with mock.patch.object(dcu, "_run", return_value="hy-smi: command not found"):
            self.assertIsNone(dcu.current_sclk())
            self.assertIsNone(dcu.current_sclk_detail()["min_mhz"])

    def test_busy_parses_pcts(self):
        with mock.patch.object(dcu, "_run", side_effect=lambda a, timeout=15.0: SAMPLE_U):
            self.assertEqual(dcu._busy_pcts(), [3.0, 97.5])
            self.assertTrue(dcu.is_busy())
            self.assertFalse(dcu.is_busy(threshold=99.0))

    def test_busy_when_unreadable(self):
        """读不到占用率按忙处理：宁可不跑。"""
        with mock.patch.object(dcu, "_run", return_value=""):
            self.assertTrue(dcu.is_busy())

    def test_perf_level_parses(self):
        with mock.patch.object(dcu, "_run", side_effect=lambda a, timeout=15.0: SAMPLE_P):
            self.assertEqual(dcu._perf_levels(), ["auto", "manual"])


class TestBackupParsing(unittest.TestCase):
    def test_reads_perflevel_from_backup(self):
        import tempfile
        import os
        text = "card\t0\n\tclocks\n\t\tsclk\t0\n\tperflevel\tauto\n"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            self.assertEqual(dcu._backup_perf_level(path), "auto")
        finally:
            os.unlink(path)

    def test_missing_file_returns_none(self):
        self.assertIsNone(dcu._backup_perf_level("/nonexistent/x.txt"))

    def test_save_refuses_existing_target(self):
        """hy-smi --save 不覆盖已存在文件，且退出码仍为 0 —— _save_clocks 必须兜住。"""
        import tempfile
        import os
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("")
            path = f.name
        try:
            # 假 hy-smi：目标存在就不写，且返回成功字样
            def fake_run(args, timeout=15.0):
                if args and args[0] == "--save":
                    return "/tmp/x already exists. Settings not saved\n"
                return ""
            with mock.patch.object(dcu, "_run", side_effect=fake_run):
                self.assertFalse(dcu._save_clocks(path))
        finally:
            # _save_clocks 会先删掉已存在目标（这正是被测行为），所以这里可能已不存在
            import contextlib
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)


class TestDoctorRender(unittest.TestCase):
    def _table(self, **levels):
        base = {
            name: {"ok": lv == "ok", "level": lv, "value": "v", "fix_hint": "", "title": t}
            for name, t in doctor.CHECKS
            for lv in [levels.get(name, "ok")]
        }
        return base

    def test_all_ok(self):
        out = doctor.render(self._table())
        self.assertIn("可以测", out)
        self.assertIn("8 项：8 ok", out)

    def test_warn_keeps_measurable(self):
        out = doctor.render(self._table(constants="warn"))
        self.assertIn("可以测，但结论要降置信度", out)

    def test_fail_blocks(self):
        out = doctor.render(self._table(gpu_free="fail"))
        self.assertIn("不能测", out)

    def test_hint_shown_only_for_non_ok(self):
        t = self._table()
        t["sclk"]["fix_hint"] = "ramp 一下"
        self.assertNotIn("ramp 一下", doctor.render(t))
        t["sclk"]["level"] = "warn"
        self.assertIn("ramp 一下", doctor.render(t))

    def test_single_check_exception_does_not_kill_table(self):
        """单项炸掉不该让整张表消失。这里把 _RUNNERS 整体替换，不碰真实 GPU。"""
        ok_stub = lambda: {"ok": True, "level": "ok", "value": "ok", "fix_hint": ""}  # noqa: E731
        runners = {name: ok_stub for name in doctor._RUNNERS}
        runners["constants"] = mock.Mock(side_effect=RuntimeError("boom"))
        with mock.patch.dict(doctor._RUNNERS, runners, clear=True):
            table = doctor.run()
        self.assertEqual(table["constants"]["level"], "fail")
        self.assertIn("boom", table["constants"]["value"])
        self.assertEqual(len(table), len(doctor.CHECKS))

    def test_exit_code(self):
        import contextlib
        import io
        buf = io.StringIO()
        with mock.patch.object(doctor, "run", return_value=self._table(gpu_free="fail")), \
             contextlib.redirect_stdout(buf):
            self.assertEqual(doctor.main([]), 1)
        with mock.patch.object(doctor, "run", return_value=self._table(constants="warn")), \
             contextlib.redirect_stdout(buf):
            self.assertEqual(doctor.main([]), 0)


class TestGpuLock(unittest.TestCase):
    def test_acquire_and_release(self):
        import os
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "gpu.lock")
        with dcu.gpu_lock(path) as a:
            self.assertTrue(a["acquired"])
            self.assertTrue(os.path.exists(path))
        with dcu.gpu_lock(path) as b:
            self.assertTrue(b["acquired"])

    def test_second_holder_times_out(self):
        import os
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "gpu.lock")
        with dcu.gpu_lock(path):
            with dcu.gpu_lock(path, wait_s=0) as second:
                self.assertFalse(second["acquired"])


if __name__ == "__main__":
    unittest.main()
