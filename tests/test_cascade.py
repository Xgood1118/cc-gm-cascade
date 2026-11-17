# -*- coding: utf-8 -*-
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cascade as mod  # noqa: E402
cascade = mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_hex(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


# ---------- 1. 分割：区间无缝 ----------

def test_split_covers_range():
    cs = cascade.split_cascades(0.5, 80.0, 4)
    assert len(cs) == 4
    assert abs(cs[0].near - 0.5) < 1e-9
    assert abs(cs[-1].far - 80.0) < 1e-9
    for a, b in zip(cs, cs[1:]):
        assert abs(a.far - b.near) < 1e-9


def test_split_no_gap_no_overlap_and_sorted():
    cs = cascade.split_cascades(1.0, 1000.0, 6)
    bounds = [cs[0].near] + [c.far for c in cs]
    # 严格单调递增：既不重叠也不留空隙（相邻端点相等由上一测试保证）
    for lo, hi in zip(bounds, bounds[1:]):
        assert hi > lo
    # 每级内部近 < 远
    for c in cs:
        assert c.far > c.near


def test_split_endpoints_exact():
    near, far = 0.5, 80.0
    cs = cascade.split_cascades(near, far, 4)
    assert cs[0].near == near
    assert cs[-1].far == far
    for a, b in zip(cs, cs[1:]):
        assert a.far == b.near


# ---------- 1b. 精度逐级递减 ----------

def test_precision_decreases_per_level():
    cs = cascade.split_cascades(0.5, 80.0, 4)
    precisions = [cascade.world_precision(c) for c in cs]
    for p in precisions:
        assert p > 0.0
    for lo, hi in zip(precisions, precisions[1:]):
        assert hi > lo  # 纹素粒度变大 = 世界空间精度逐级递远（变粗糙）


def test_fixed_resolution():
    cs = cascade.split_cascades(0.5, 80.0, 4)
    assert {c.resolution for c in cs} == {1024}
    for c in cs:
        assert math.isclose(
            c.texel_world, (c.far - c.near) / c.resolution, rel_tol=1e-15
        )


# ---------- 2. 衔接：权重和为 1 ----------

def test_blend_sums_to_one():
    w = cascade.blend_region(0.0, 1.0, 0.5)
    assert abs(sum(w) - 1.0) < 1e-9


def test_blend_sums_to_one_across_world_space_band():
    cs = cascade.split_cascades(0.5, 80.0, 4)
    for c in cs[:-1]:
        start, end, samples = cascade.blend_band(c)
        assert start < c.far < end
        for depth, wa, wb in samples:
            assert abs(wa + wb - 1.0) < 1e-12
    # 边界点：两端权重确定（0/1），中间为 0.5/0.5
    assert cascade.blend_region(2.0, 4.0, 2.0) == (1.0, 0.0)
    assert cascade.blend_region(2.0, 4.0, 4.0) == (0.0, 1.0)
    mid = cascade.blend_region(2.0, 4.0, 3.0)
    assert abs(mid[0] - 0.5) < 1e-12 and abs(mid[1] - 0.5) < 1e-12


def test_blend_smoothstep_derivative_zero_at_ends():
    # smoothstep 在 u=0 与 u=1 处导数为 0：相邻差分在两端最小，无硬边
    a, b = 0.0, 10.0
    eps = 1e-6
    d0 = abs(cascade.blend_region(a, b, a + eps)[1] - 0.0) / eps
    d1 = abs(1.0 - cascade.blend_region(a, b, b - eps)[1]) / eps
    dm = abs(
        cascade.blend_region(a, b, 5.0 + eps)[1]
        - cascade.blend_region(a, b, 5.0 - eps)[1]
    ) / (2 * eps)
    assert d0 < dm
    assert d1 < dm


def test_blend_zero_width_region_is_deterministic():
    assert cascade.blend_region(3.0, 3.0, 2.9) == (1.0, 0.0)
    assert cascade.blend_region(3.0, 3.0, 3.0) == (0.0, 1.0)
    assert sum(cascade.blend_region(3.0, 3.0, 9.0)) == 1.0


# ---------- 2b. 纹素对齐 ----------

def test_tighten_bounds_snaps_to_texel_grid():
    cs = cascade.split_cascades(0.5, 80.0, 4)
    snapped = cascade.tighten_bounds(cs, {"resolution": 1024})
    assert len(snapped) == len(cs)
    for c, s in zip(cs, snapped):
        # 吸附后仍完整覆盖原区间
        assert s.near <= c.near
        assert s.far >= c.far
        # 近/远平面落在整数个纹素上（残余偏差仅为浮点舍入）
        res = cascade.alignment_error(s)
        assert abs(res[0]) < 1e-9
        assert abs(res[1]) < 1e-9
        # 方向性吸附：近平面 floor（偏差 [0,1) 纹素），远平面 ceil（偏差 (-1,0]）
        assert 0.0 <= s.align_error[0] < 1.0
        assert -1.0 < s.align_error[1] <= 0.0


def test_tighten_bounds_respects_origin():
    cs = cascade.split_cascades(0.5, 80.0, 3)
    origin = 0.25
    snapped = cascade.tighten_bounds(cs, {"origin": origin, "resolution": 1024})
    for s in snapped:
        res = cascade.alignment_error(s, origin=origin)
        assert abs(res[0]) < 1e-9 and abs(res[1]) < 1e-9
        # 显式核验：(bound - origin) / texel 是整数
        near_cells = (s.near - origin) / s.texel_world
        far_cells = (s.far - origin) / s.texel_world
        assert abs(near_cells - round(near_cells)) < 1e-9
        assert abs(far_cells - round(far_cells)) < 1e-9


def test_tighten_bounds_does_not_mutate_input():
    cs = cascade.split_cascades(0.5, 80.0, 4)
    before = [(c.near, c.far) for c in cs]
    cascade.tighten_bounds(cs, None)
    assert [(c.near, c.far) for c in cs] == before


# ---------- 3. 边界：级数变化 / 极小近平面 / 极大远平面 ----------

def test_level_count_variations():
    for levels in range(1, 7):
        cs = cascade.split_cascades(0.5, 80.0, levels)
        assert len(cs) == levels
        assert cs[0].near == 0.5
        assert cs[-1].far == 80.0
        for a, b in zip(cs, cs[1:]):
            assert a.far == b.near
            assert b.near > a.near


def test_single_level_covers_whole_range():
    cs = cascade.split_cascades(0.5, 80.0, 1)
    assert len(cs) == 1
    assert cs[0].near == 0.5 and cs[0].far == 80.0


def test_tiny_near_plane():
    tiny = 1e-6
    cs = cascade.split_cascades(tiny, 100.0, 4)
    assert cs[0].near == tiny
    assert cs[-1].far == 100.0
    bounds = [cs[0].near] + [c.far for c in cs]
    assert all(hi > lo for lo, hi in zip(bounds, bounds[1:]))
    snapped = cascade.tighten_bounds(cs, {"resolution": 1024})
    for s in snapped:
        res = cascade.alignment_error(s)
        assert abs(res[0]) < 1e-9 and abs(res[1]) < 1e-9


def test_huge_far_plane():
    huge = 1e12
    cs = cascade.split_cascades(1.0, huge, 5)
    assert cs[0].near == 1.0
    assert cs[-1].far == huge
    bounds = [cs[0].near] + [c.far for c in cs]
    assert all(hi > lo for lo, hi in zip(bounds, bounds[1:]))
    snapped = cascade.tighten_bounds(cs, {"resolution": 1024})
    for s in snapped:
        res = cascade.alignment_error(s)
        # 数值尺度大，按相对容差核验网格对齐
        assert abs(res[0]) < 1e-9 and abs(res[1]) < 1e-9
        assert s.near <= cs[s.index].near
        assert s.far >= cs[s.index].far


def test_invalid_inputs_raise():
    import pytest

    with pytest.raises(ValueError):
        cascade.split_cascades(0.0, 80.0, 4)
    with pytest.raises(ValueError):
        cascade.split_cascades(1.0, 1.0, 4)
    with pytest.raises(ValueError):
        cascade.split_cascades(2.0, 1.0, 4)
    with pytest.raises(ValueError):
        cascade.split_cascades(0.5, 80.0, 0)
    with pytest.raises(ValueError):
        cascade.split_cascades(0.5, 80.0, -1)
    with pytest.raises(ValueError):
        cascade.split_cascades(float("nan"), 80.0, 4)
    with pytest.raises(ValueError):
        cascade.split_cascades(0.5, float("inf"), 4)


# ---------- 4. 输出确定性：两次一致 ----------

def test_split_is_deterministic():
    a = cascade.split_cascades(0.5, 80.0, 4)
    b = cascade.split_cascades(0.5, 80.0, 4)
    assert [(c.near, c.far, c.resolution) for c in a] == [
        (c.near, c.far, c.resolution) for c in b
    ]


def test_cli_output_is_deterministic(tmp_path):
    out1 = tmp_path / "c1.json"
    out2 = tmp_path / "c2.json"
    data = os.path.join(ROOT, "data", "split.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    for out in (out1, out2):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "cascade.cli",
                data,
                "--levels",
                "4",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    payload = json.loads(out1.read_text(encoding="utf-8"))
    assert payload["levels"] == 4
    assert len(payload["cascades"]) == 4
