"""经验矩阵虚拟博弈对照
固定零和矩阵[[0.8,-0.2],[-0.4,0.6]]，每步均匀采一个矩阵元素、观测Gaussian噪声σ=0.2，经验均值初值0。解析minimax或虚拟博弈只用已学习矩阵；真实矩阵仅测量Nash间隙，不实现一般Markov博弈。
按旧对手频率同时选行最大和列最小响应，再累积响应计数。
时序、边界与未覆盖范围见核心注释及 docs/extended-classic.md。
"""
import math
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "fictitious_play_reference",
    "name": "经验矩阵虚拟博弈对照",
    "family": "extended_classic",
    "coverage_refs": [],
    "implementation_kind": "component_experiment",
    "task": "extended_empirical_zero_sum_game",
    "baseline": "minimax_2x2",
    "metric": "真实矩阵 Nash 间隙",
    "unit": "payoff",
    "higher_better": False,
    "budget": "payoff_samples",
    "scope": "component-empirical-matrix-game",
    "chapter_paths": [
        "foundations/deep/multi-agent/"
    ],
    "sources": [
        "https://link.springer.com/article/10.1007/BF01448847"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "固定零和矩阵[[0.8,-0.2],[-0.4,0.6]]，每步均匀采一个矩阵元素、观测Gaussian噪声σ=0.2，经验均值初值0。解析minimax或虚拟博弈只用已学习矩阵；真实矩阵仅测量Nash间隙，不实现一般Markov博弈。",
    "question": "对经验对手频率的最佳响应怎样逼近零和混合策略？",
    "limitations": "仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。"
}

def update(matrix,counts,i,j,payoff):
    counts[i][j]+=1
    matrix[i][j]+=(payoff-matrix[i][j])/counts[i][j]

def minimax(matrix):
    # 行玩家最大化两条仿射函数的下包络；端点与可行交点足够。
    a,b=matrix[0]
    c,d=matrix[1]
    candidates=[0.,1.]
    denominator=a-c-b+d
    if denominator!=0:
        crossing=(d-c)/denominator
        if 0<=crossing<=1:
            candidates.append(crossing)
    value=lambda p:min(p*a+(1-p)*c,p*b+(1-p)*d)
    p=max(candidates,key=value)
    return [p,1-p],value(p)

def gap(matrix,row,column):
    best_row=max(dot(line,column) for line in matrix)
    best_column=min(sum(row[i]*matrix[i][j] for i in range(2)) for j in range(2))
    return best_row-best_column

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,matrix,counts=random.Random(seed),[],[[0.,0.],[0.,0.]],[[0,0],[0,0]]
    truth=[[.8,-.2],[-.4,.6]]
    row_counts,column_counts=[1.,1.],[1.,1.]
    for t in range(1,steps+1):
        i,j=rng.randrange(2),rng.randrange(2)
        payoff=truth[i][j]+rng.gauss(0,.2)
        update(matrix,counts,i,j,payoff)
        old_row=[v/sum(row_counts) for v in row_counts]
        old_column=[v/sum(column_counts) for v in column_counts]
        br=max(range(2),key=lambda i:dot(matrix[i],old_column))
        bc=min(range(2),key=lambda j:sum(old_row[i]*matrix[i][j] for i in range(2)))
        row_counts[br]+=1
        column_counts[bc]+=1
        row=[v/sum(row_counts) for v in row_counts]
        column=[v/sum(column_counts) for v in column_counts]
        log(rows,t,gap(truth,row,column),steps,emit,row_probability=row[0],column_probability=column[0])
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
