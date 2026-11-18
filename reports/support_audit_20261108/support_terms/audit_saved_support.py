"""Read-only audit of all saved C/D model components; no model forward calls.

Checkpoint tensors are deserialized only to verify metadata/configuration. No
flow class is imported, constructed or evaluated. Only aggregate outputs are
written. This post-hoc audit is conditional on recorded rounded arrays.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, localcontext, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

COMPONENTS = {"log_p_y", "log_q_z", "base_value", "base_logpdf", "spline_logabsdet", "log_support_cdf", "log_s", "z", "b", "log_cdf_y", "log_sf_y"}
QUANTILES = [0., .000001, .00001, .0001, .001, .01, .05, .5, .95, .99, 1.]
ULP_THRESHOLDS = [1, 2, 4, 8, 16, 32, 64, 128, 1024]

def utc(): return datetime.now(timezone.utc).isoformat()
def load(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()
def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
def under(root, rel):
    result = (root / rel).resolve()
    if not result.is_relative_to(root.resolve()): raise ValueError("path escaped root")
    return result
def verify(root, rec):
    p = under(root, rec["path"])
    if p.stat().st_size != rec["bytes"] or sha(p) != rec["sha256"]: raise ValueError(f"artifact mismatch: {p}")
    return p
def numbers(a):
    x = np.asarray(a)
    finite = np.isfinite(x)
    good = x[finite].astype(np.float64)
    out = {"count": len(x), "finite": int(finite.sum()), "nan": int(np.isnan(x).sum()), "positive_infinity": int(np.isposinf(x).sum()), "negative_infinity": int(np.isneginf(x).sum()), "zero": int((x == 0).sum()), "negative_zero": int(((x == 0) & np.signbit(x)).sum()), "positive_zero": int(((x == 0) & ~np.signbit(x)).sum()), "nonzero": int((x != 0).sum())}
    if len(good):
        lo, hi = float(good.min()), float(good.max())
        qs = [lo] * len(QUANTILES) if lo == hi else np.quantile(good, QUANTILES).tolist()
        out.update(minimum=lo, maximum=hi, quantiles={str(q): float(v) for q, v in zip(QUANTILES, qs)})
    else: out.update(minimum=None, maximum=None, quantiles={})
    return out

def tail_certificate(bmin, bmax):
    """Mathematical log-domain upper bounds, without computing exp(-b).

    For b>0, t=exp(-b), excluded mass=t/(1+t) and |log F|=log1p(t).
    For either quantity a: -b-t <= log(a) <= -b. This bounds mathematical
    values at the saved rounded b, not unrecorded infinite-precision inputs.
    """
    assert 0 < bmin <= bmax
    with localcontext() as ctx:
        ctx.prec = 90
        ln10 = Decimal(10).ln()
        upper = -Decimal.from_float(float(bmin)) / ln10
        lower_anchor = -Decimal.from_float(float(bmax)) / ln10
        rounded_upper = upper.to_integral_value(rounding=ROUND_CEILING)
        half32 = Decimal(150) * Decimal(2).ln()
        half64 = Decimal(1075) * Decimal(2).ln()
        return {"basis": "saved float32 b, exact mathematical logistic tail; no exp(-b) evaluated",
                "log10_maximum_upper_bound": str(upper), "log10_minimum_anchor": str(lower_anchor),
                "conservative_maximum_bound": f"both excluded mass and |log(F(b))| < 10^({rounded_upper})",
                "log_value_bound": "-b-exp(-b) <= log(excluded mass), log(|log(F(b))|) <= -b",
                "log10_anchor_error_upper_bound": "exp(-b)/ln(10)",
                "all_b_beyond_float32_half_subnormal_threshold": Decimal.from_float(float(bmin)) > half32,
                "all_b_beyond_float64_half_subnormal_threshold": Decimal.from_float(float(bmin)) > half64,
                "float32_half_subnormal_b_threshold": str(half32), "float64_half_subnormal_b_threshold": str(half64)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, research, output = args.repo.resolve(), args.research.resolve(), args.output.resolve()
    if output.exists(): raise ValueError("output directory must be new")
    output.mkdir(parents=True)
    start = time.perf_counter()
    torch.set_num_threads(1)
    roots = {"core": repo / "reports/correction_v2_20261107/study", "full": repo / "reports/corrected_formal_extension_20261107/study", "information": repo / "reports/information_extension_20261108/study"}
    score_root = research / "fresh_scores"
    scoring = load(score_root / "scoring_summary.json")
    assert scoring["status"] == "COMPLETE_FROZEN_240_CD_SCORES" and not scoring["failures"]
    union_file = research / "FROZEN_FRESH_MODEL_UNION.json"
    assert sha(union_file) == scoring["union_sha256"]
    union = load(union_file)
    assert union["status"] == "COMPLETE_FROZEN_FRESH_MODEL_UNION" and union["expected_models"] == 240
    roster_file = verify(research, union["predetermined_roster"])
    assert sha(roster_file) == scoring["roster_sha256"]
    roster = load(roster_file)
    roster_ids = {x["model_id"] for x in roster["models"]}
    models = {x["model_id"]: x for x in union["models"]}
    assert len(models) == 240 and set(models) == roster_ids
    assert Counter(x["cohort"] for x in models.values()) == {"information80": 80, "legacy160": 160}
    sources = []
    for rel, digest in scoring["source_files"].items():
        p = under(repo, rel)
        assert sha(p) == digest, rel
        sources.append({"path": rel, "sha256": digest})
    engine = (repo / "src/representation_flow/engine.py").read_text(encoding="utf-8")
    assert "SPLINE_BOUNDS = 8.0" in engine
    assert "output = torch.where(inside, output, value)" in engine
    assert "return -torch.nn.functional.softplus(-value)" in engine
    assert "base_upper, _ = self.transform_to_base(upper, context)" in engine
    configs = {}
    input_bindings = [{"path": str(score_root / "scoring_summary.json"), "sha256": sha(score_root / "scoring_summary.json")}, {"path": str(union_file), "sha256": sha(union_file)}, {"path": str(roster_file), "sha256": sha(roster_file)}]
    for mid, row in models.items():
        ck = verify(roots[row["root_key"]], row["artifacts"]["checkpoint"])
        run = verify(roots[row["root_key"]], row["artifacts"]["run_manifest"])
        payload = torch.load(ck, map_location="cpu", weights_only=True)
        config = payload["model_config"]
        assert config["bound"] == 8.0
        assert payload["experiment_identity"] == row["experiment_identity"]
        assert payload["seed"] == row["labels"]["seed"]
        configs[mid] = {"model_config": config, "checkpoint_sha256": row["artifacts"]["checkpoint"]["sha256"], "run_manifest_sha256": row["artifacts"]["run_manifest"]["sha256"]}
        del payload
    print("Verified all 240 checkpoint identities/configs without constructing or evaluating models", flush=True)
    all_models, block_records, checked_arrays = [], [], []
    for record in scoring["catalogs"]:
        block = record["block_id"]
        assert block in ("C", "D")
        manifest_file = under(score_root, record["path"])
        assert sha(manifest_file) == record["sha256"]
        manifest = load(manifest_file)
        assert manifest["status"] == "COMPLETE_FROZEN_240_CATALOG_SCORES" and not manifest["failures"]
        assert manifest["union_sha256"] == scoring["union_sha256"]
        assert manifest["source_files"] == scoring["source_files"]
        assert {x["model_id"] for x in manifest["models"]} == roster_ids and len(manifest["models"]) == 240
        assert manifest["launched_events"] == 500000
        assert sum(manifest["launched_counts"].values()) == 500000
        count = manifest["eligible_events"]
        assert sum(manifest["eligible_counts"].values()) == count
        block_root = manifest_file.parent
        input_bindings.append({"path": str(manifest_file), "sha256": sha(manifest_file)})
        catalog = {}
        for name, array_record in manifest["arrays"].items():
            p = verify(block_root, array_record)
            arr = np.load(p, mmap_mode="r", allow_pickle=False)
            assert list(arr.shape) == array_record["shape"] == [count]
            assert str(arr.dtype) == array_record["dtype"]
            checked_arrays.append({"block": block, "model_id": "CATALOG", "name": name, **array_record})
            catalog[name] = arr
        uid_list = catalog["event_uid"].tolist()
        uid_digest = canonical_hash(uid_list)
        assert len(set(uid_list)) == count and uid_digest == manifest["event_uid_sha256"]
        del uid_list
        assert Counter(int(x) for x in catalog["group_id"]) == {int(k): v for k, v in manifest["eligible_counts"].items()}
        p32, y32 = np.asarray(catalog["p_in"], np.float32), np.asarray(catalog["y"], np.float32)
        valid = y32 < p32
        assert np.isfinite(p32).all() and np.isfinite(y32).all()
        input_margin = p32.astype(np.float64) - y32.astype(np.float64)
        block_records.append({"block": block, "launched_events": manifest["launched_events"], "eligible_events": count, "source_excluded_by_predeclared_eligibility": manifest["launched_events"] - count, "eligible_uid_sha256": uid_digest, "events_sha256": manifest["events_sha256"], "counts_sha256": manifest["counts_sha256"], "input_y32_below_p32": int(valid.sum()), "input_y32_equal_p32": int((y32 == p32).sum()), "input_y32_above_p32": int((y32 > p32).sum()), "p32_minus_y32_GeV_c": numbers(input_margin), "catalog_arrays_sha_shape_dtype_verified": True})
        for index, row in enumerate(manifest["models"]):
            mid = row["model_id"]
            frozen = models[mid]
            assert row["status"] == "COMPLETE_FROZEN_MODEL_SCORE"
            for key in ["cohort", "checkpoint_kind", "root_key", "labels", "experiment_identity"]:
                assert row[key] == frozen[key], (mid, key)
            assert row["checkpoint_sha256"] == configs[mid]["checkpoint_sha256"]
            assert row["eligible_events"] == count and row["event_uid_sha256"] == uid_digest
            assert set(row["arrays"]) == COMPONENTS
            local_manifest = block_root / "models" / mid / "model_manifest.json"
            assert load(local_manifest) == row
            values, component_counts = {}, {}
            for name, array_record in row["arrays"].items():
                p = verify(block_root, array_record)
                arr = np.load(p, mmap_mode="r", allow_pickle=False)
                assert list(arr.shape) == array_record["shape"] == [count]
                assert str(arr.dtype) == array_record["dtype"] == "float32"
                checked_arrays.append({"block": block, "model_id": mid, "name": name, **array_record})
                values[name] = arr
                component_counts[name] = {"finite": int(np.isfinite(arr).sum()), "nan": int(np.isnan(arr).sum()), "positive_infinity": int(np.isposinf(arr).sum()), "negative_infinity": int(np.isneginf(arr).sum())}
            b, z = values["b"], values["z"]
            margin = b.astype(np.float64) - z.astype(np.float64)
            ulp = b.astype(np.float64) - np.nextafter(b, np.float32(-np.inf)).astype(np.float64)
            relative = margin / ulp
            support = numbers(values["log_support_cdf"])
            outside = int((b > configs[mid]["model_config"]["bound"]).sum())
            cert = tail_certificate(float(b.min()), float(b.max())) if outside == count else None
            nll = float(np.mean(-values["log_p_y"], dtype=np.float64))
            assert nll == row["nll"], (mid, nll, row["nll"])
            base = values["base_value"]
            # Saved base(z) is sufficient for a further ordering check. No base(b)
            # is saved; identity at b follows from config/source + b>bound.
            base_margin = b.astype(np.float64) - base.astype(np.float64)
            model = {"block": block, "model_id": mid, "cohort": row["cohort"], "labels": row["labels"], "event_count": count, "event_uid_sha256": uid_digest, **configs[mid], "model_manifest_sha256": sha(local_manifest), "all_component_hash_shape_dtype_checks": True, "component_finiteness": component_counts, "support_logcdf": support, "b": numbers(b), "z": numbers(z), "b_minus_z_from_saved_float32_in_float64": numbers(margin), "b_minus_base_z_from_saved_float32_in_float64": numbers(base_margin), "standardized_margin_downward_ULP_of_b": numbers(relative), "standardized_small_positive_ulp_counts": {str(k): int(((relative > 0) & (relative <= k)).sum()) for k in ULP_THRESHOLDS}, "z_equal_b": int((z == b).sum()), "z_above_b": int((z > b).sum()), "b_above_identity_tail_bound": outside, "identity_tail_base_b_equals_b": outside == count, "logp_nonfinite": count - component_counts["log_p_y"]["finite"], "logp_negative_infinity": component_counts["log_p_y"]["negative_infinity"], "nll_recomputed": nll, "same_order_all_eligible_rows_saved": True, "no_post_model_array_drop": len(values["log_p_y"]) == count, "mathematical_tail_certificate": cert}
            all_models.append(model)
            if (index + 1) % 40 == 0:
                print(f"{block}: verified {index+1}/240 complete saved component sets", flush=True)
            del values, margin, relative, b, z, base, base_margin
    assert {x["block"] for x in block_records} == {"C", "D"}
    cohorts = []
    for block in ["C", "D"]:
        for cohort in ["information80", "legacy160", "all_registered240"]:
            selected = [x for x in all_models if x["block"] == block and (x["cohort"] == cohort or cohort == "all_registered240")]
            assert len(selected) == {"information80": 80, "legacy160": 160, "all_registered240": 240}[cohort]
            bmin = min(x["b"]["minimum"] for x in selected)
            bmax = max(x["b"]["maximum"] for x in selected)
            cohorts.append({"block": block, "cohort": cohort, "registered_model_count": len(selected), "model_event_evaluations": sum(x["event_count"] for x in selected), "counting_scope": "repeated model-event evaluations, not independent events or predictors", "support_logcdf_zero": sum(x["support_logcdf"]["zero"] for x in selected), "support_logcdf_negative_zero": sum(x["support_logcdf"]["negative_zero"] for x in selected), "support_logcdf_positive_zero": sum(x["support_logcdf"]["positive_zero"] for x in selected), "support_logcdf_nonzero": sum(x["support_logcdf"]["nonzero"] for x in selected), "support_logcdf_nonfinite": sum(x["event_count"]-x["support_logcdf"]["finite"] for x in selected), "b_minimum": bmin, "b_maximum": bmax, "b_minus_z_minimum": min(x["b_minus_z_from_saved_float32_in_float64"]["minimum"] for x in selected), "b_minus_z_maximum": max(x["b_minus_z_from_saved_float32_in_float64"]["maximum"] for x in selected), "minimum_margin_ULP_b": min(x["standardized_margin_downward_ULP_of_b"]["minimum"] for x in selected), "z_equal_b": sum(x["z_equal_b"] for x in selected), "z_above_b": sum(x["z_above_b"] for x in selected), "logp_nonfinite": sum(x["logp_nonfinite"] for x in selected), "all_b_in_identity_tail": all(x["identity_tail_base_b_equals_b"] for x in selected), "mathematical_tail_certificate": tail_certificate(bmin,bmax) if all(x["identity_tail_base_b_equals_b"] for x in selected) else None})
    duplicate_file = research / "full20_rerun_identity_review.json"
    duplicate = load(duplicate_file)
    assert duplicate["status"] == "COMPLETE_FULL20_RERUN_IDENTITY_COMPARISON" and len(duplicate["pairs"]) == 20
    by_key = {(x["block"], x["model_id"]): x for x in all_models}
    array_index = {(x["block"], x["model_id"], x["name"]): x["sha256"] for x in checked_arrays}
    dup_checks = []
    for pair in duplicate["pairs"]:
        for block in ["C", "D"]:
            legacy, new = pair["legacy_run_id"], pair["new_run_id"]
            equal = {name: array_index[(block,legacy,name)] == array_index[(block,new,name)] for name in COMPONENTS}
            assert all(equal.values())
            dup_checks.append({"block":block,"earlier_run_id":legacy,"new_run_id":new,"all_eleven_component_files_byte_identical": True})
    input_bindings.append({"path": str(duplicate_file), "sha256": sha(duplicate_file)})
    result = {"schema": "saved-fresh-support-audit/v1", "status": "COMPLETE_SAVED_480_MODEL_SUPPORT_AUDIT", "generated_utc": utc(), "elapsed_seconds": time.perf_counter()-start, "analysis_intent": "post_hoc_read_only_to_known_C_D", "new_training_or_checkpoint_inference": False, "checkpoint_read_scope": "weights_only deserialization to inspect model_config and identity; no flow instantiated or evaluated", "coverage": {"catalogs":2,"registered_models_per_catalog":240,"new_primary_per_catalog":80,"earlier_per_catalog":160,"registered_model_catalog_pairs":480,"model_component_arrays_verified":len(checked_arrays)-12,"catalog_arrays_verified":12,"duplicate_Full_pairs_per_catalog":20}, "catalogs":block_records,"cohort_summaries":cohorts,"models":all_models,"duplicate_full_component_checks":dup_checks,"input_bindings":input_bindings,"frozen_sources_verified":sources,"precision_and_formula": {"support":"log(F(b))=-softplus(-base(b))", "original_logpdf":"log(q(z))-log(s)-log(F(b)) with y<p_in support mask", "identity_tail":"Each of two RQS layers returns its input exactly for value>8. All registered configs use bound=8; hence recorded b>8 implies base(b)=b without evaluating conditioner nets.", "zero_interpretation":"Observed zero is a saved floating-point value. Mathematical logistic tail is strictly positive; log-domain upper bounds avoid inventing a representable nonzero probability.", "margin_definition":"Exact float64 difference of two stored float32 values b-z; it does not recover unrecorded infinite-precision preprocessing.", "no_drop_scope":"All eligible UID rows are saved for each model and all logp are checked. Eligibility exclusions precede scoring and are not numerical drops."},"limitations":["No new model evaluation or float64 refit/replay was performed.","The tail certificate concerns the mathematical logistic function at the saved rounded b.","No uniform guarantee for other events, models, devices, or numerical precision is inferred.","Repeated model-event counts and the two catalogs are not independent optimizer replications; all 20 exact Full rerun pairs are disclosed.","Raw ledger/order reconciliation is an independent companion audit; this audit binds its saved catalog UID hash and component files."]}
    save(output/"support_audit.json", result)
    save(output/"verified_array_manifest.json", {"schema":"saved-support-array-verification/v1","status":"ALL_HASH_SHAPE_DTYPE_CHECKS_PASSED","arrays":checked_arrays})
    csv_rows=[]
    for x in all_models:
        csv_rows.append({"block":x["block"],"model_id":x["model_id"],"cohort":x["cohort"],**x["labels"],"events":x["event_count"],"event_uid_sha256":x["event_uid_sha256"],"logcdf_min":x["support_logcdf"]["minimum"],"logcdf_max":x["support_logcdf"]["maximum"],"logcdf_zero":x["support_logcdf"]["zero"],"logcdf_negative_zero":x["support_logcdf"]["negative_zero"],"logcdf_nonzero":x["support_logcdf"]["nonzero"],"logcdf_nonfinite":x["event_count"]-x["support_logcdf"]["finite"],"b_min":x["b"]["minimum"],"b_max":x["b"]["maximum"],"z_min":x["z"]["minimum"],"z_max":x["z"]["maximum"],"margin_min":x["b_minus_z_from_saved_float32_in_float64"]["minimum"],"margin_max":x["b_minus_z_from_saved_float32_in_float64"]["maximum"],"margin_min_ULP_b":x["standardized_margin_downward_ULP_of_b"]["minimum"],"z_equal_b":x["z_equal_b"],"z_above_b":x["z_above_b"],"logp_nonfinite":x["logp_nonfinite"],"all_b_gt8":x["identity_tail_base_b_equals_b"],"excluded_mass_log10_upper_bound":x["mathematical_tail_certificate"]["log10_maximum_upper_bound"] if x["mathematical_tail_certificate"] else ""})
    with (output/"model_support_summary.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    outputs=[p for p in output.iterdir() if p.is_file()]
    save(output/"MANIFEST.json",{"schema":"saved-support-audit-delivery/v1","status":"COMPLETE","generated_utc":utc(),"script_sha256":sha(Path(__file__)),"inputs":input_bindings,"outputs":[{"path":p.name,"sha256":sha(p),"bytes":p.stat().st_size} for p in outputs],"only_aggregate_outputs":True})
    print(json.dumps({"status":result["status"],"elapsed_seconds":result["elapsed_seconds"],"cohorts":cohorts}),flush=True)

if __name__=="__main__":main()
