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

    @staticmethod
    def _fake_hy_smi(g_outputs, perf="auto"):
        """按命令分派假 hy-smi：-g 依次吐 g_outputs，-p 固定。"""
        seq = iter(g_outputs)

        def run(args, timeout=15.0):
            if args and args[0] == "-g":
                return next(seq, g_outputs[-1])
            if args and args[0] == "-p":
                return f"HCU[0]\t\t: Performance Level: {perf}\n"
            return ""

        return run

    def test_keeps_stable_low_card(self):
        with mock.patch.object(dcu, "_run", side_effect=self._fake_hy_smi([SAMPLE_G] * 3)), \
             mock.patch.object(dcu.time, "sleep", return_value=None):
            d = dcu.current_sclk_detail(samples=3)
        self.assertEqual(d["levels"], [5, 8])  # card0 稳定 5，card1 稳定 8

    def test_mode_beats_single_spike(self):
        """负载刚结束时的单次抖动（1200）不该被读成"在降频"。"""
        spikes = [
            "HCU[0]: sclk clock level: 8 (1350Mhz)\n",
            "HCU[0]: sclk clock level: 5 (1200Mhz)\n",
            "HCU[0]: sclk clock level: 8 (1350Mhz)\n",
        ]
        with mock.patch.object(dcu, "_run", side_effect=self._fake_hy_smi(spikes)), \
             mock.patch.object(dcu.time, "sleep", return_value=None):
            d = dcu.current_sclk_detail(samples=3)
        self.assertEqual(d["min_level"], 8)

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


class TestFreeDevicesVramGate(unittest.TestCase):
    """空闲卡判据必须**同时**看占用率和可用显存。

    ## 事故（2026-11-26 凌晨，本机实测）

    8 张卡 HCU 全是 0%，`free_devices()` 报"8 张都空闲"。但卡 0–3
    各只剩 4.9–6.9 GB —— 被**容器外**别的租户占着（本容器 `ps` 里只有
    17 个进程，全是 shell，`hy-smi` 的 VRAM% 却是 90%+）。

    `--parallel-cards` 照着这个列表派活，第一道 mHC 的 M=16384 档
    （参考实现峰值 4.9 GB）立刻 OOM。而 OOM 在报告里长得像
    "这个臂的 kernel 有问题"，实际是调度器选错卡 —— 会污染结论。

    所以两条都要卡：忙的不用，**显存不够的也不用**。
    """

    def _pcts(self, vals):
        return mock.patch.object(dcu, "_busy_pcts", return_value=vals)

    def test_idle_but_low_vram_excluded(self):
        # 编号 0/1 空闲但只有 5 GB，2 空闲且 100 GB
        with self._pcts([0.0, 0.0, 0.0]), \
             mock.patch.object(dcu, "_free_vram_gb",
                               return_value={0: 5.0, 1: 5.0, 2: 100.0}):
            self.assertEqual(dcu.free_devices(), [2])

    def test_busy_high_vram_excluded(self):
        """显存再多，正在算也不用——抢卡会同时毁掉两边的测量。"""
        with self._pcts([95.0, 0.0]), \
             mock.patch.object(dcu, "_free_vram_gb",
                               return_value={0: 130.0, 1: 50.0}):
            self.assertEqual(dcu.free_devices(), [1])

    def test_unknown_vram_does_not_exclude(self):
        """读不到显存时按"未知"放行，否则无 torch 环境一张卡都选不出来。"""
        with self._pcts([0.0, 0.0]), \
             mock.patch.object(dcu, "_free_vram_gb", return_value={}):
            self.assertEqual(dcu.free_devices(), [0, 1])

    def test_threshold_is_respected(self):
        with self._pcts([0.0]), \
             mock.patch.object(dcu, "_free_vram_gb", return_value={0: 50.0}):
            self.assertEqual(dcu.free_devices(min_free_gb=60.0), [])
            self.assertEqual(dcu.free_devices(min_free_gb=40.0), [0])

    def test_pick_device_prefers_most_vram_not_lowest_index(self):
        """按编号取会稳定选中别人占剩的那张（卡 0）。"""
        with self._pcts([0.0, 0.0, 0.0]), \
             mock.patch.object(dcu, "_free_vram_gb",
                               return_value={0: 5.5, 1: 6.9, 2: 139.0}), \
             mock.patch("torch.cuda.set_device"):
            self.assertEqual(dcu.pick_device(), 2)

    def test_pick_device_returns_none_when_nothing_usable(self):
        with self._pcts([99.0]), \
             mock.patch.object(dcu, "_free_vram_gb", return_value={0: 200.0}):
            self.assertIsNone(dcu.pick_device())


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
