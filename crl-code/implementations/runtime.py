"""Algorithm-independent experiment IO. Standard library only.

算法的环境交互、目标计算、梯度和更新都留在算法文件中。
这里仅负责参数、逐阶段日志、完整种子记录和可复查的绘图。
曲线显示真实运行的均值及 ±1 个训练种子标准差，不是置信区间。
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def discover():
    """Read literal META dictionaries without importing optional torch dependencies."""
    result = {}
    for path in sorted(ROOT.glob('*/*.py')):
        if path.name.startswith('_'):
            continue
        meta = metadata(path)
        if meta:
            key = meta['id']
            if key in result:
                raise ValueError('Duplicate algorithm ID: '+key)
            result[key] = path
    return result


def metadata(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'META' for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def load(algorithm_id):
    paths = discover()
    if algorithm_id not in paths:
        raise ValueError('Unknown algorithm '+algorithm_id)
    path = paths[algorithm_id]
    # Package names avoid collisions between the groups' _common modules.
    if str(ROOT.parent) not in sys.path:
        sys.path.insert(0, str(ROOT.parent))
    name = 'implementations.'+path.parent.name+'.'+path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*.py'))}


def comparable(metas):
    """A shared ordinate is insufficient: task and budget must also match."""
    for key in ('family', 'task', 'metric', 'unit', 'higher_better', 'budget'):
        values = [m.get(key) for m in metas]
        if any(v is None for v in values) or any(v != values[0] for v in values[1:]):
            raise ValueError('Incompatible comparison '+key+': '+repr(values))


def validate_rows(rows, steps):
    if not rows:
        raise ValueError('Algorithm returned no observations')
    previous = -1
    for row in rows:
        x, y = row.get('step'), row.get('value')
        if isinstance(x, bool) or not isinstance(x, int) or x <= previous or x < 0 or x > steps:
            raise ValueError('step must increase within declared budget: '+repr(row))
        if isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y):
            raise ValueError('Nonfinite or missing metric: '+repr(row))
        if not isinstance(row.get('phase'), str):
            raise ValueError('Each observation needs a phase label')
        previous = x
    if rows[-1]['step'] != steps:
        raise ValueError('Algorithm did not reach the registered horizon')


def summarize(records):
    """Do not impute missing runs. A failed comparison is not silently averaged away."""
    if any(record['status'] != 'complete' for record in records):
        raise ValueError('Incomplete population cannot be summarized as completed curves')
    curves = {}
    for record in records:
        if record['status'] != 'complete':
            continue
        curves.setdefault(record['algorithm'], []).append(record['rows'])
    result = {}
    for algorithm, runs in curves.items():
        grids = [[r['step'] for r in rows] for rows in runs]
        if any(grid != grids[0] for grid in grids[1:]):
            raise ValueError('Seeds have different observation clocks: '+algorithm)
        result[algorithm] = [
            {'step': x, 'mean': statistics.mean(values),
             'std': statistics.stdev(values) if len(values) > 1 else 0.0,
             'n': len(values)}
            for i, x in enumerate(grids[0])
            for values in [[rows[i]['value'] for rows in runs]]
        ]
    return result


def svg_plot(curves, metas, width=900, height=390):
    """SVG from observed rows only. Stable colors + line patterns for accessibility."""
    all_points = [p for curve in curves.values() for p in curve]
    if not all_points:
        return '<svg xmlns="http://www.w3.org/2000/svg"><text y="20">No completed runs.</text></svg>'
    names = {m['id']: m['name'] for m in metas}
    margin = (85, 30, 75, 65)  # left, right, top, bottom
    left, right, top, bottom = margin
    xmax = max(p['step'] for p in all_points) or 1
    lo = min(p['mean']-p['std'] for p in all_points)
    hi = max(p['mean']+p['std'] for p in all_points)
    pad = .08*(hi-lo) if hi > lo else max(1, abs(hi)*.1)
    lo, hi = lo-pad, hi+pad
    X = lambda v: left+v/xmax*(width-left-right)
    Y = lambda v: height-bottom-(v-lo)/(hi-lo)*(height-top-bottom)
    esc = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="chart-title chart-desc">',
             '<title id="chart-title">'+esc(metas[0]['metric'])+'</title>',
             '<desc id="chart-desc">Actual runs; line = mean, band = plus/minus one sample standard deviation across training seeds. Not a confidence interval.</desc>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="system-ui,sans-serif" font-size="12" fill="#243349">']
    for j in range(5):
        val = lo+(hi-lo)*j/4
        y = Y(val)
        parts += [f'<path d="M{left},{y:.2f}H{width-right}" stroke="#e3e8ef"/>',
                  f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end">{val:.3g}</text>']
    for j in range(6):
        val = xmax*j/5
        parts.append(f'<text x="{X(val):.2f}" y="{height-bottom+22}" text-anchor="middle">{val:g}</text>')
    colors = ['#2563a5', '#b46114', '#765898', '#457552', '#b44473']
    for index, (algorithm, points) in enumerate(curves.items()):
        color = colors[index % len(colors)]
        dash = ['none', '8 4', '3 3', '9 3 2 3', '2 5'][index % 5]
        edge = [(X(p['step']), Y(p['mean']-p['std'])) for p in points]
        edge += [(X(p['step']), Y(p['mean']+p['std'])) for p in reversed(points)]
        parts.append('<polygon fill="'+color+'" opacity="0.12" points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in edge)+'"/>')
        line = ' '.join(f'{X(p["step"]):.2f},{Y(p["mean"]):.2f}' for p in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}" points="{line}"/>')
        lx, ly = left+(index % 3)*260, 22+(index//3)*22
        parts.append(f'<path d="M{lx},{ly}h24" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/><text x="{lx+31}" y="{ly+4}">{esc(names[algorithm])}</text>')
    parts += [f'<path d="M{left},{top}V{height-bottom}H{width-right}" fill="none" stroke="#526075"/>',
              f'<text x="{(left+width-right)/2}" y="{height-14}" text-anchor="middle">{esc(metas[0]["budget"])}</text>',
              f'<text transform="translate(18,{(top+height-bottom)/2}) rotate(-90)" text-anchor="middle">{esc(metas[0]["unit"])}</text>', '</g></svg>']
    return ''.join(parts)


def experiment(ids, seeds, steps, out):
    if not ids or len(set(ids)) != len(ids) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide unique algorithms and training seeds')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be positive')
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError('seeds must be nonnegative integers')
    paths = discover()
    metas = [metadata(paths[a]) for a in ids]
    comparable(metas)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous run.
    start = time.monotonic()
    manifest = {'schema_version': 1, 'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'python': platform.python_version(), 'algorithms': metas, 'seeds': seeds, 'steps': steps,
                'source_sha256': source_hashes(), 'runs': [], 'scope': 'Teaching experiment; not paper-scale reproduction',
                'uncertainty': 'Mean ± 1 sample standard deviation across independent training seeds; not a confidence interval'}
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    save()
    # Preserve the selected source and shared algorithm dependencies once per group.
    for name, expected in manifest['source_sha256'].items():
        data = (ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Source changed while freezing run snapshot: '+name)
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    records = []
    for algorithm in ids:
        for seed in seeds:
            run_dir = out/algorithm/('seed-'+str(seed))
            run_dir.mkdir(parents=True)
            print(f'[prepare] algorithm={algorithm} seed={seed} budget={steps}', flush=True)
            record = {'algorithm': algorithm, 'seed': seed, 'status': 'running'}
            manifest['runs'].append(record)
            (run_dir/'config.json').write_text(json.dumps({'algorithm': algorithm, 'seed': seed, 'steps': steps,
                'meta': metadata(paths[algorithm]), 'source_sha256': manifest['source_sha256']}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            save()
            with (run_dir/'events.jsonl').open('w', encoding='utf-8') as events:
                last_print = [-1]
                emitted = []
                def emit(row):
                    encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
                    # Persist before validation so a malformed event remains inspectable.
                    events.write(encoded+'\n')
                    events.flush()
                    emitted.append(json.loads(encoded))
                    x, y = row.get('step'), row.get('value')
                    if isinstance(x, bool) or not isinstance(x, int) or x < 0 or x > steps or (len(emitted)>1 and x <= emitted[-2]['step']):
                        raise ValueError('Invalid emitted event clock')
                    if isinstance(y, bool) or not isinstance(y, (int,float)) or not math.isfinite(y) or not isinstance(row.get('phase'),str):
                        raise ValueError('Invalid emitted metric or phase')
                    decile = int(10*row['step']/steps)
                    if decile != last_print[0]:
                        print(f'[{row["phase"]}] {algorithm} seed={seed} {row["step"]}/{steps} value={row["value"]:.6g}', flush=True)
                        last_print[0] = decile
                try:
                    module = load(algorithm)
                    rows = module.run(seed=seed, steps=steps, emit=emit)
                    validate_rows(rows, steps)
                    if not emitted:
                        for row in rows:
                            emit(row)
                    if rows != emitted:
                        raise ValueError('Returned rows differ from raw emitted evidence')
                    fields = ['step', 'value', 'phase']+sorted(set().union(*(r.keys() for r in rows))-{'step','value','phase'})
                    with (run_dir/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
                    record.update(status='complete', final=rows[-1]['value'], observations=len(rows))
                    records.append({**record, 'rows': rows})
                except Exception as exc:
                    record.update(status='failed', error=type(exc).__name__+': '+str(exc))
                    records.append(dict(record))
                    print('[failed] '+record['error'], file=sys.stderr, flush=True)
            save()
    failed = [r for r in records if r['status'] != 'complete']
    drift = manifest['source_sha256'] != source_hashes()
    if drift:
        manifest['source_drift'] = 'Source changed during execution; retain evidence, rerun a new version'
    # Incomplete populations retain raw evidence but never get a success-looking plot.
    if not failed and not drift:
        curves = summarize(records)
        (out/'curves.json').write_text(json.dumps(curves, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        (out/'learning-curves.svg').write_text(svg_plot(curves, metas), encoding='utf-8')
        (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Algorithm comparison</title><style>body{max-width:1000px;margin:40px auto;padding:20px;font:16px/1.7 system-ui;color:#243349}img{width:100%}code{overflow-wrap:anywhere}</style><h1>'+html.escape(metas[0]['metric'])+'</h1><p>真实教学运行。相同任务、预算与种子列表。线为种子均值，阴影为 ±1 个样本标准差，不是置信区间。结果不证明在其他任务上的优势。</p><img src="learning-curves.svg" alt="Learning curves"><p><a href="manifest.json">配置、源码摘要和全部运行</a> · <a href="curves.json">绘图数据</a></p>', encoding='utf-8')
    manifest['status'] = 'failed' if failed or drift else 'complete'
    manifest['elapsed_seconds'] = time.monotonic()-start
    manifest['artifacts'] = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    save()
    print('[complete]' if manifest['status']=='complete' else '[incomplete]', str(out), flush=True)
    return manifest


def run_cli(meta, run):
    parser = argparse.ArgumentParser(description=meta['description'])
    parser.add_argument('--steps', type=int, default=meta.get('defaults', {}).get('steps', 1000))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--baseline', default=meta.get('baseline'), help='Algorithm ID, or none for a single method')
    parser.add_argument('--out', type=Path, required=True, help='A new output directory; existing results are never overwritten')
    args = parser.parse_args()
    ids = [meta['id']]
    if args.baseline and args.baseline != 'none' and args.baseline not in ids:
        ids.append(args.baseline)
    result = experiment(ids, args.seeds, args.steps, args.out)
    if result['status'] != 'complete':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('algorithms', nargs='*')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.list:
        for algorithm, path in discover().items():
            m = metadata(path)
            print(algorithm+'\t'+m['name']+'\t'+m.get('task',''))
        return
    if not args.algorithms or args.out is None:
        parser.error('Use --list or provide algorithm IDs and --out')
    if experiment(args.algorithms, args.seeds, args.steps, args.out)['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
