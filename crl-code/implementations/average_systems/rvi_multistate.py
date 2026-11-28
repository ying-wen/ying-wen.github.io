"""RVI · known three-state model
已知三状态双动作模型，同步Bellman最优备份后减去参考状态0值；与同模型不减常数的VI对照。每扫描6个状态动作备份。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "rvi_multistate",
    "name": "RVI · known three-state model",
    "family": "average_systems",
    "implementation_kind": "teaching_method",
    "coverage_refs": [],
    "task": "average_known_model_planning",
    "baseline": "unnormalized_vi_multistate",
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
    "description": "已知三状态双动作模型，同步Bellman最优备份后减去参考状态0值；与同模型不减常数的VI对照。每扫描6个状态动作备份。",
    "question": "没有折扣压缩时，为什么需要去除公共价值平移？",
    "limitations": "已知小模型、精确期望备份、遍历非周期任务。种子不影响确定性结果，不把模型扫描当真实交互。普通RVI在其他周期任务中可能振荡。"
}

def update(h):
    # All right-hand sides read OLD h; not an in-place Gauss-Seidel sweep.
    backed = [max(REWARDS[s][a]+sum(P[s][a][sp]*h[sp] for sp in range(3))
                  for a in range(2)) for s in range(3)]
    offset = backed[0]
    return [v-offset for v in backed], offset


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
        rate = reference
        log(rows, t, steps, emit, **prediction_diagnostics(h, rate, true_rate, true_bias),
            model_backups=6*t, environment_updates=0)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)

