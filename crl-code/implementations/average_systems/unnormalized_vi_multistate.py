"""Unanchored VI · offset comparison
同已知模型Bellman备份，不去除公共增长。相对值仍可正确，绝对值线性长大；展示规范选择，不假装其必然控制失败。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "unnormalized_vi_multistate",
    "name": "Unanchored VI · offset comparison",
    "family": "average_systems",
    "implementation_kind": "component_experiment",
    "coverage_refs": [],
    "task": "average_known_model_planning",
    "baseline": "rvi_multistate",
    "metric": "最优偏差参考对齐RMSE",
    "unit": "value",
    "higher_better": False,
    "budget": "model_sweeps",
    "scope": "known-model-control-planning",
    "chapter_paths": [
        "algorithms/average-reward/",
        "construction/planning/"
    ],
    "sources": [
        "https://proceedings.mlr.press/v139/wan21a.html"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "同已知模型Bellman备份，不去除公共增长。相对值仍可正确，绝对值线性长大；展示规范选择，不假装其必然控制失败。",
    "question": "没有折扣压缩时，为什么需要去除公共价值平移？",
    "limitations": "已知小模型、精确期望备份、遍历非周期任务。种子不影响确定性结果，不把模型扫描当真实交互。普通RVI在其他周期任务中可能振荡。"
}

def update(h):
    # All right-hand sides read OLD h; not an in-place Gauss-Seidel sweep.
    backed = [max(REWARDS[s][a]+sum(P[s][a][sp]*h[sp] for sp in range(3))
                  for a in range(2)) for s in range(3)]
    offset = backed[0]
    return backed, offset


def run(seed=0, steps=1200, emit=None):
    check_steps(steps)
    rows, h = [], [0., 0., 0.]
    true_rate, true_bias, _ = optimal()
    log(rows, 0, steps, emit, **prediction_diagnostics(h, 0., true_rate, true_bias),
        model_backups=0, environment_updates=0)
    for t in range(1, steps+1):
        old = h
        h, reference = update(old)
        # Unanchored reference itself diverges; its increment estimates g.
        rate = reference-old[0]
        log(rows, t, steps, emit, **prediction_diagnostics(h, rate, true_rate, true_bias),
            model_backups=6*t, environment_updates=0)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)

