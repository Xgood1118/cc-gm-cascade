# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cascade as mod  # noqa: E402


def test_split_covers_range():
    cs = cascade.split_cascades(0.5, 80.0, 4)
    assert len(cs) == 4
    assert abs(cs[0].near - 0.5) < 1e-9
    assert abs(cs[-1].far - 80.0) < 1e-9
    for a, b in zip(cs, cs[1:]):
        assert abs(a.far - b.near) < 1e-9


def test_blend_sums_to_one():
    w = cascade.blend_region(0.0, 1.0, 0.5)
    assert abs(sum(w) - 1.0) < 1e-9
