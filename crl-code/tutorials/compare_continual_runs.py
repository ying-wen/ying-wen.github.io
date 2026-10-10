"""Read-only recomputation of the pinned 1200-step integrated-ring records.

Usage:
    python3 compare_continual_runs.py --archive raw-runs.zip

Uses only the Python standard library. Reads ZIP members in memory, never
extracts them and never imports or executes archived producer code. This is
one specific recorded protocol, not a generic RL CSV reader or a training
entrypoint. Output is JSON on stdout; invalid evidence exits with no report.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import zipfile


METHODS = ("integrated_recent_model", "integrated_cumulative_model",
           "integrated_no_planning")
SEEDS = list(range(5))
STEPS, SWITCH, WINDOW = 1200, 600, 100
GRID = [1] + list(range(20, STEPS + 1, 20))
PROTOCOL = "integrated-continuing-ring-1200-v1"
MAX_MEMBERS, MAX_MEMBER_BYTES, MAX_TOTAL_BYTES = 2000, 32*1024*1024, 64*1024*1024
PINNED_SOURCES = {
    "integrated_agents/_system.py": "190f6947d32ec6547bde8ddaf629e997d5e8c6ab2b926d1902dab1fcf5d992e0",
    "integrated_agents/integrated_recent_model.py": "fbc844c6331c711fc26ded040547fcc30d1e6214023062cd3a3f5531809e17d2",
    "integrated_agents/integrated_cumulative_model.py": "99e055287029fd92417240c1cb59e372711042300425d4257f1892dba8194c1b",
    "integrated_agents/integrated_no_planning.py": "0547a911f88b420cabfe89f14c4ca10e06d306203d228886525dc424ad5dc259",
    "runtime.py": "99b835e09a8edb20e8db3145de1013ee78a9e5d67a675e67997be76fbdba7295",
}
META_FIELDS = {
    "family": "integrated_agents", "task": "integrated_continuing_ring",
    "metric": "最近100个真实步的平均外部奖励",
    "unit": "reward_per_environment_step", "budget": "environment_steps",
    "higher_better": True,
}
HEADER = ["step", "value", "phase", "average_reward", "cap_finishes",
          "estimated_rate", "invalidations", "lifetime_reward", "model_backups",
          "model_cells", "model_error_samples", "model_reward_mae",
          "option_finishes", "pending_duration", "real_backups", "subtask_backups"]
COUNTS = {"step", "cap_finishes", "invalidations", "model_backups", "model_cells",
          "model_error_samples", "option_finishes", "pending_duration",
          "real_backups", "subtask_backups"}
MONOTONE = {"cap_finishes", "invalidations", "model_backups", "option_finishes",
            "real_backups", "subtask_backups"}


class EvidenceError(ValueError):
    """The archive cannot support the declared protocol and comparison."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def near(actual, expected, context):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
            f"{context}: {actual!r} differs from {expected!r}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    require(isinstance(name, str) and name and "\\" not in name,
            "Invalid ZIP member path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            f"Noncanonical ZIP member path: {name}")


def json_object(data, context):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{context}: duplicate JSON key {key}")
            result[key] = value
        return result

    def constant(value):
        raise EvidenceError(f"{context}: nonfinite JSON number {value}")

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{context}: invalid JSON") from error

    def finite(node):
        if isinstance(node, float):
            require(math.isfinite(node), f"{context}: nonfinite JSON number")
        elif isinstance(node, dict):
            for child in node.values():
                finite(child)
        elif isinstance(node, list):
            for child in node:
                finite(child)
    finite(value)
    require(isinstance(value, dict), f"{context}: expected JSON object")
    return value


def read_archive(path):
    try:
        require(Path(path).stat().st_size <= MAX_TOTAL_BYTES, "Archive exceeds protocol size allowance")
        archive_bytes = Path(path).read_bytes()
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            infos = archive.infolist()
            require(len(infos) <= MAX_MEMBERS, "Too many archive members for this protocol")
            require(all(not info.is_dir() and not (info.flag_bits & 1) for info in infos),
                    "Protocol archive must contain unencrypted files only")
            require(all(info.file_size <= MAX_MEMBER_BYTES for info in infos)
                    and sum(info.file_size for info in infos) <= MAX_TOTAL_BYTES,
                    "Uncompressed archive exceeds protocol size allowance")
            names = [info.filename for info in infos]
            require(len(names) == len(set(names)), "Duplicate ZIP member path")
            for name in names:
                safe_name(name)
            members = {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as error:
        raise EvidenceError(f"Cannot read intact ZIP archive: {error}") from error
    require("manifest.json" in members, "Missing manifest.json")
    manifest = json_object(members["manifest.json"], "manifest.json")
    require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
            "Unsupported manifest schema")
    require(manifest.get("status") == "complete", "Manifest does not declare complete runs")
    require(type(manifest.get("steps")) is int and manifest["steps"] == STEPS
            and manifest.get("seeds") == SEEDS
            and all(type(seed) is int for seed in manifest["seeds"]),
            "Unsupported protocol: require 1200 steps and seeds 0..4")
    artifacts, sources = manifest.get("artifacts"), manifest.get("source_sha256")
    require(isinstance(artifacts, dict) and isinstance(sources, dict),
            "Manifest lacks artifact/source digests")
    require(set(members) == set(artifacts) | {"manifest.json"},
            "Missing or unregistered archive artifact")
    for name, expected in artifacts.items():
        safe_name(name)
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                f"Invalid artifact digest: {name}")
        require(digest(members[name]) == expected, f"Artifact SHA256 mismatch: {name}")
    for name, expected in sources.items():
        safe_name(name)
        require("source/" + name in members, f"Missing frozen source: {name}")
        require(digest(members["source/" + name]) == expected,
                f"Source SHA256 mismatch: {name}")
    for name, expected in PINNED_SOURCES.items():
        require(sources.get(name) == expected,
                f"Unsupported producer protocol SHA256: {name}")
    return members, manifest, digest(archive_bytes)


def parse_csv(data, context):
    try:
        reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
        require(next(reader, None) == HEADER, f"{context}: wrong CSV schema")
        rows = []
        for cells in reader:
            require(len(cells) == len(HEADER), f"{context}: malformed CSV row")
            row = {}
            for key, cell in zip(HEADER, cells):
                if key == "phase":
                    row[key] = cell
                elif key in COUNTS:
                    require(re.fullmatch(r"0|[1-9][0-9]*", cell) is not None,
                            f"{context}: invalid integer {key}")
                    row[key] = int(cell)
                else:
                    try:
                        row[key] = float(cell)
                    except ValueError as error:
                        raise EvidenceError(f"{context}: invalid number {key}") from error
                    require(math.isfinite(row[key]), f"{context}: nonfinite {key}")
            rows.append(row)
    except (UnicodeError, csv.Error) as error:
        raise EvidenceError(f"{context}: unreadable CSV") from error
    require([row["step"] for row in rows] == GRID,
            f"{context}: require all 61 protocol checkpoints")
    return rows


def validate_rows(rows, method, context):
    by_step = {row["step"]: row for row in rows}
    previous = dict.fromkeys(COUNTS | {"lifetime_reward"}, 0)
    for row in rows:
        step = row["step"]
        require(row["phase"] == ("before-change" if step <= SWITCH else "after-change"),
                f"{context}: phase differs from change after step600")
        near(row["average_reward"] * step, row["lifetime_reward"], context + " cumulative rate")
        require(-.02-1e-9 <= row["value"] <= .98+1e-9, context + ": reward window infeasible")
        arrivals = (row["value"] + .02) * min(step, WINDOW)
        near(arrivals, round(arrivals), context + " window reward support")
        if step <= WINDOW:
            near(row["value"], row["lifetime_reward"] / step, context + " initial window")
        elif step-WINDOW in by_step:
            near(row["value"], (row["lifetime_reward"]-by_step[step-WINDOW]["lifetime_reward"]) / WINDOW,
                 context + " sliding window")
        interval = step-previous["step"]
        arrivals = row["lifetime_reward"]-previous["lifetime_reward"]+.02*interval
        near(arrivals, round(arrivals), context + " interval reward support")
        require(-1e-9 <= arrivals <= interval+1e-9, context + ": impossible interval reward")
        for key in MONOTONE:
            require(row[key] >= previous[key], f"{context}: nonmonotone {key}")
        require(row["real_backups"]-previous["real_backups"] <= interval,
                context + ": too many completed segments")
        planning = 0 if method == "integrated_no_planning" else 4
        require(row["model_backups"] == planning*row["real_backups"],
                context + ": model backup clock differs")
        require(row["subtask_backups"] == 2*step, context + ": subtask backup clock differs")
        require(row["real_backups"] <= step and row["cap_finishes"] <= row["option_finishes"]
                <= row["real_backups"] and row["pending_duration"] < 6,
                context + ": invalid completed/pending segment counts")
        require(row["real_backups"] <= step-row["pending_duration"]
                <= 6*row["real_backups"], context + ": completed duration budget differs")
        require(row["model_cells"] <= 26 and row["invalidations"] <= 2*row["real_backups"],
                context + ": invalid model/version counts")
        require(0 <= row["model_error_samples"] <= min(100, row["real_backups"])
                and row["model_reward_mae"] >= 0, context + ": invalid conditional model error")
        if not row["model_error_samples"]:
            near(row["model_reward_mae"], 0, context + " empty model-error history")
        previous = row
    end = by_step[STEPS]
    near(end["value"], (end["lifetime_reward"]-by_step[STEPS-WINDOW]["lifetime_reward"])/WINDOW,
         context + " final100 window")
    return {
        "readings": {
            "lifetime_reward": end["lifetime_reward"],
            "full_rate": end["lifetime_reward"]/STEPS,
            "before_change_rate": by_step[SWITCH]["lifetime_reward"]/SWITCH,
            "after_change_rate": (end["lifetime_reward"]-by_step[SWITCH]["lifetime_reward"])/(STEPS-SWITCH),
            "final_window_rate": (end["lifetime_reward"]-by_step[STEPS-WINDOW]["lifetime_reward"])/WINDOW,
        },
        "costs": {
            "environment_steps": STEPS, "real_backups": end["real_backups"],
            "model_backups": end["model_backups"], "subtask_backups": end["subtask_backups"],
            "total_backup_writes": end["real_backups"]+end["model_backups"]+end["subtask_backups"],
        },
        "pending_duration": end["pending_duration"],
    }


def summarize(runs):
    def summary(values):
        return {"n": len(values), "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values)}
    return {group: {key: summary([run[group][key] for run in runs])
                    for key in runs[0][group]} for group in ("readings", "costs")}


def recompute(path):
    members, manifest, archive_hash = read_archive(path)
    metas = manifest.get("algorithms")
    require(isinstance(metas, list) and all(isinstance(meta, dict) for meta in metas),
            "Manifest lacks algorithm metadata")
    require(all(isinstance(meta.get("id"), str) for meta in metas), "Invalid algorithm ID")
    require(len(metas) == len({meta["id"] for meta in metas}), "Duplicate algorithm metadata")
    metadata = {meta["id"]: meta for meta in metas}
    receipts = manifest.get("runs")
    require(isinstance(receipts, list) and all(isinstance(run, dict) for run in receipts),
            "Manifest lacks run receipts")
    require(all(isinstance(run.get("algorithm"), str) and type(run.get("seed")) is int
                and run.get("status") == "complete" for run in receipts),
            "Incomplete or invalid run receipt")
    keys = [(run["algorithm"], run["seed"]) for run in receipts]
    require(len(keys) == len(set(keys)) and set(keys) == {(method, seed) for method in metadata for seed in SEEDS},
            "Duplicate, missing or unexpected method/seed receipt")
    receipt_map = dict(zip(keys, receipts))
    runs = []
    for method in METHODS:
        require(method in metadata, f"Missing required method: {method}")
        meta = metadata[method]
        require(all(meta.get(key) == value and type(meta.get(key)) is type(value)
                    for key, value in META_FIELDS.items()), f"Wrong protocol metadata: {method}")
        # Static literal metadata only. Never import or execute this source.
        tree = ast.parse(members["source/integrated_agents/"+method+".py"].decode("utf-8"))
        literal = [node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "META" for target in node.targets)]
        require(len(literal) == 1 and ast.literal_eval(literal[0]) == meta,
                f"Manifest metadata differs from pinned producer: {method}")
        for seed in SEEDS:
            prefix = f"{method}/seed-{seed}/"
            for name in ("config.json", "metrics.csv", "events.jsonl"):
                require(prefix+name in members, f"Missing run member: {prefix+name}")
            config = json_object(members[prefix+"config.json"], prefix+"config.json")
            require(config.get("algorithm") == method and type(config.get("seed")) is int
                    and config["seed"] == seed and type(config.get("steps")) is int
                    and config["steps"] == STEPS and config.get("meta") == meta
                    and config.get("source_sha256") == manifest["source_sha256"],
                    f"Config method/seed/steps/metadata/source mismatch: {prefix}")
            rows = parse_csv(members[prefix+"metrics.csv"], prefix+"metrics.csv")
            receipt = receipt_map[method, seed]
            require(type(receipt.get("observations")) is int and receipt["observations"] == len(GRID),
                    f"Wrong observation receipt: {prefix}")
            require(type(receipt.get("final")) in (int, float), f"Missing final receipt: {prefix}")
            near(receipt["final"], rows[-1]["value"], prefix+" final receipt")
            events = [json_object(line, prefix+"events.jsonl") for line in members[prefix+"events.jsonl"].splitlines()]
            require(all(set(event) == set(HEADER)
                        and all(type(event[key]) is int for key in COUNTS)
                        and type(event["phase"]) is str
                        and all(type(event[key]) in (int, float) for key in set(HEADER)-COUNTS-{"phase"})
                        for event in events), f"Wrong event schema/types: {prefix}")
            require(events == rows, f"Events differ from CSV evidence: {prefix}")
            metrics = validate_rows(rows, method, prefix)
            runs.append({"method": method, "seed": seed, **metrics,
                         "config_sha256": digest(members[prefix+"config.json"]),
                         "csv_sha256": digest(members[prefix+"metrics.csv"])})
    methods = {method: summarize([run for run in runs if run["method"] == method]) for method in METHODS}
    lookup = {(run["method"], run["seed"]): run for run in runs}
    paired = {}
    for other, label in ((METHODS[1], "recent_minus_cumulative"), (METHODS[2], "recent_minus_no_planning")):
        differences = []
        for seed in SEEDS:
            left, right = lookup[METHODS[0], seed], lookup[other, seed]
            differences.append({"seed": seed, **{group: {key: left[group][key]-right[group][key]
                                for key in left[group]} for group in ("readings", "costs")}})
        paired[label] = {"left": METHODS[0], "right": other, "per_seed": differences,
                         **summarize(differences)}
    return {
        "schema_version": 1, "protocol_id": PROTOCOL, "status": "validated_existing_records",
        "archive_sha256": archive_hash,
        "manifest_sha256": digest(members["manifest.json"]),
        "validated_artifacts": len(manifest["artifacts"]),
        "validated_sources": len(manifest["source_sha256"]),
        "protocol": {"steps": STEPS, "change_after_step": SWITCH, "window": WINDOW,
                     "checkpoints_per_run": len(GRID), "seeds": SEEDS,
                     "methods": list(METHODS), "pinned_source_sha256": PINNED_SOURCES},
        "definitions": {
            "hash_validation": "Artifact/source SHA256 checks establish package internal consistency and a pinned producer version, not an author signature or source authentication. Compare archive_sha256 with a separately trusted receipt to identify exact original bytes; rewriting data and manifest hashes does not authenticate the observations.",
            "lifetime_reward": "Sum of actual external rewards on steps1..1200, including pending tails.",
            "full_rate": "Lifetime external reward / 1200 environment steps.",
            "before_change_rate": "External reward on steps1..600 / 600.",
            "after_change_rate": "External reward on steps601..1200 / 600.",
            "final_window_rate": "External reward on steps1101..1200 / 100.",
            "costs": "Counts of completed high-level real/model writes and two subtask writes per environment step; total_backup_writes sums these three counts. Different update kinds do not have equivalent FLOP cost; this does not measure wall time or all model maintenance.",
            "sample_sd": "Sample standard deviation across five complete run seeds (denominator n-1), not a confidence interval.",
            "paired_differences": "Subtract recorded readings/costs for matching seed IDs, then average those five differences; no significance test or general causal superiority claim.",
        },
        "runs": runs, "methods": methods, "paired_differences": paired,
        "interpretation": [
            "This is an existing teaching component experiment, not a complete STOMP/OaK replication or new training.",
            "The same seed gives the same external uniform stream, but action-dependent paths differ. Internal action and planning draws share one agent RNG; removing planning changes that RNG path.",
            "Cumulative models average samples within each surviving cell; option-policy version changes still invalidate corresponding cells and high-level values.",
            "Pending-tail rewards count towards actual return; unfinished segments do not create complete high-level backups or model samples.",
            "Whole-lifetime and final100-window rankings may differ; model_reward_mae uses only at most100 previously modeled complete segments and is not full-space model accuracy.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path,
                        help="Existing raw-runs.zip for the pinned 1200-step protocol")
    args = parser.parse_args(argv)
    try:
        report = recompute(args.archive)
    except (EvidenceError, KeyError, TypeError, SyntaxError, OverflowError, RecursionError) as error:
        print(f"Evidence rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
