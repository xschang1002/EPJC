"""Disambiguate two valid margin arithmetic definitions and join read-only audits.

No model components are recomputed. Only catalog p/y are read for an additional
float32 subtraction distribution. The original audit outputs are preserved.
"""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from audit_saved_support import numbers, load, save, sha, utc, verify

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
RESEARCH = Path(r"E:\JINST REP\_research_extension_20261108")
RESULT = HERE / "results"

def main():
    before = RESULT / "before_margin_scope_clarification"
    if before.exists(): raise ValueError("clarification already applied; retain its receipt")
    before.mkdir()
    for name in ["support_audit.json", "MANIFEST.json"]:
        shutil.copy2(RESULT/name, before/name)
    summary = load(RESULT / "support_audit.json")
    original_models_hash = hashlib.sha256(json.dumps(summary["models"],sort_keys=True).encode()).hexdigest()
    ledger_path = AUDIT / "input_margin_review.json"
    replay_path = AUDIT / "standardization_replay.json"
    ledger, replay = load(ledger_path), load(replay_path)
    assert ledger["status"] == "PASS_ORIGINAL_LEDGER_CACHE_AND_FLOAT32_INPUT_SUPPORT"
    assert replay["status"] == "PASS_ALL_SAVED_Z_B_BITWISE_REPLAY"
    assert replay["input_margin_review_sha256"] == sha(ledger_path)
    assert replay["registered_model_blocks_compared"] == 480 and replay["total_bitwise_mismatches"] == 0
    assert replay["component_values_compared"] == 479308800
    assert replay["rebuilt_z_ge_b_model_event_pairs"] == 0
    ledger_blocks = {x["block_id"]:x for x in ledger["blocks"]}
    reconciled = []
    for block in summary["catalogs"]:
        label = block["block"]
        peer = ledger_blocks[label]
        root = RESEARCH / "fresh_scores" / label
        manifest = load(root/"score_manifest.json")
        assert sha(root/"score_manifest.json") == peer["score_manifest_sha256"]
        assert block["eligible_uid_sha256"] == peer["event_uid_order_sha256"]
        for key in ["events_sha256","launched_events","eligible_events"]:
            assert block[key] == peer[key]
        arrays = {}
        for name in ["p_in", "y"]:
            path = verify(root, manifest["arrays"][name])
            arrays[name] = np.load(path,allow_pickle=False).astype(np.float32)
        actual = numbers(arrays["p_in"] - arrays["y"])
        peer_q = {float(k):v for k,v in peer["margin32_GeV_c"]["quantiles"].items()}
        assert {float(k):v for k,v in actual["quantiles"].items()} == peer_q
        old = block.pop("p32_minus_y32_GeV_c")
        block["exact_float64_difference_of_float32_inputs_GeV_c"] = old
        block["float32_subtraction_of_float32_inputs_GeV_c"] = actual
        block["input_margin_metric_definitions"] = {
            "exact_float64_difference_of_float32_inputs_GeV_c": "float64(p32)-float64(y32), exact difference of stored float32 input values evaluated in float64",
            "float32_subtraction_of_float32_inputs_GeV_c": "float32(p32-y32), actual float32 subtraction; matches the independent ledger audit",
            "support_predicate": "y32<p32 compares inputs directly, without subtracting them; both difference definitions are strictly positive here"}
        reconciled.append({"block":label,"eligible_events":block["eligible_events"],"uid_hash_equal":True,"events_hash_equal":True,"actual_float32_margin_quantiles_equal":True,"exact_difference_minimum":old["minimum"],"float32_subtraction_minimum":actual["minimum"]})
    assert original_models_hash == hashlib.sha256(json.dumps(summary["models"],sort_keys=True).encode()).hexdigest()
    summary["companion_reconciliation"] = {"input_margin_review_sha256":sha(ledger_path),"standardization_replay_sha256":sha(replay_path),"all_480_standardization_arrays_bitwise_replayed":True,"scope":"The companion audit reconstructs ordered z/b from frozen pure preprocessing without neural evaluation. Together these checks establish row correspondence; declared UID hashes alone would not prove it."}
    summary["precision_and_formula"]["input_margin_arithmetic"] = "Two separately labeled input metrics: float64(p32)-float64(y32) versus float32(p32-y32). The standardized margin remains float64(saved b32)-float64(saved z32)."
    save(RESULT / "support_audit.json", summary)
    reconciliation = {"schema":"independent-support-audit-reconciliation/v1","status":"PASS_LEDGER_INPUT_STANDARDIZATION_AND_COMPONENTS","generated_utc":utc(),"catalogs":reconciled,"input_margin_review":{"path":str(ledger_path),"sha256":sha(ledger_path)},"standardization_replay":{"path":str(replay_path),"sha256":sha(replay_path)},"saved_support_audit":{"path":"support_audit.json","sha256":sha(RESULT/"support_audit.json")},"registered_model_catalog_pairs":480,"standardization_arrays_bitwise_replayed":960,"standardization_values_bitwise_replayed":479308800,"standardization_bitwise_mismatches":0,"no_neural_evaluation_or_refitting":True}
    save(RESULT / "INDEPENDENT_RECONCILIATION.json", reconciliation)
    receipt = {"schema":"support-input-margin-scope-clarification/v1","status":"COMPLETE","generated_utc":utc(),"reason":"Distinguish the exact difference of float32 input values in float64 from subtraction rounded in float32; both are legitimate but numerically different metrics.","original_support_sha256":sha(before/"support_audit.json"),"revised_support_sha256":sha(RESULT/"support_audit.json"),"original_model_statistics_hash":original_models_hash,"all_model_component_statistics_unchanged":True,"original_manifest_sha256":sha(before/"MANIFEST.json"),"reconciliation_sha256":sha(RESULT/"INDEPENDENT_RECONCILIATION.json"),"additional_reads":"Only p_in/y catalog arrays and companion aggregate audit records; no 5280-component rehash, model forward, training, or simulation.","postprocessor_sha256":sha(Path(__file__))}
    save(RESULT / "MARGIN_SCOPE_CLARIFICATION.json", receipt)
    manifest = load(before/"MANIFEST.json")
    manifest["schema"] = "saved-support-audit-delivery/v2"
    manifest["generated_utc"] = utc()
    manifest["original_execution_manifest_sha256"] = sha(before/"MANIFEST.json")
    manifest["postprocessing_script_sha256"] = sha(Path(__file__))
    manifest["postprocessing_note"] = "Original script_sha256 still identifies the executed 480-model audit; the separate postprocessor clarifies input margins and joins independent ledger/standardization receipts."
    manifest["outputs"] = [{"path":p.name,"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(RESULT.iterdir()) if p.is_file() and p.name!="MANIFEST.json"]
    save(RESULT/"MANIFEST.json",manifest)
    print(json.dumps({"status":reconciliation["status"],"catalogs":reconciled}))

if __name__=="__main__": main()
