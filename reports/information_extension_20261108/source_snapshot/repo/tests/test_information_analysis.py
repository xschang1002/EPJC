"""Synthetic checks of independent planned inference, never fresh-catalog reads."""
import numpy as np
import pytest
from types import SimpleNamespace

from representation_flow.information_analysis import (
    flow_nll_components, paired_component_summary, location_scale_shape,
    pit_diagnostics, source_location_burden,
)
from representation_flow.information_uncertainty import (
    paired_seed_gains, primary_labels, planned_contrasts, practical_effect,
    stratified_paired_bootstrap,
)


def _score(n=100):
    base = np.linspace(-3, -1, n)
    determinant = np.linspace(-.5, .8, n)
    scale = np.full(n, -8.0)
    support = np.linspace(-.1, -.0001, n)
    return {"base_logpdf": base, "spline_logabsdet": determinant,
            "log_q_z": base + determinant, "log_s": scale,
            "log_support_cdf": support,
            "log_p_y": base + determinant - scale - support}


def test_density_terms_reconstruct_and_wrong_sign_or_tamper_is_rejected():
    raw = _score()
    corrected = {k: v.copy() for k, v in raw.items()}
    corrected["base_logpdf"] += .04
    corrected["spline_logabsdet"] -= .01
    corrected["log_q_z"] += .03
    corrected["log_s"] += .02
    corrected["log_support_cdf"] -= .005
    corrected["log_p_y"] += .015
    report = paired_component_summary(raw, corrected, np.arange(100) % 2)
    assert report["overall"]["nll_gain"] == pytest.approx(.015)
    assert report["overall"]["component_sum_minus_gain"] == pytest.approx(0, abs=1e-14)
    assert report["overall"]["log_support_cdf"] == pytest.approx(.005)
    corrupted = dict(corrected, log_q_z=corrected["log_q_z"] + .01)
    with pytest.raises(ValueError, match="reconstruct"):
        flow_nll_components(corrupted)


def test_launched_bootstrap_includes_eligible_mixture_and_shares_event_weights():
    ids = np.r_[np.zeros(40, dtype=int), np.ones(20, dtype=int)]
    x = np.where(ids == 0, 2.0, -1.0)
    gains = np.column_stack([x, 2*x+7])
    result = stratified_paired_bootstrap(
        gains, ids, {0:80, 1:80}, labels=["a", "b"],
        contrasts={"a":[1,0], "b":[0,1], "paired_identity":[2,-1]},
        n_resamples=500, seed=81, batch_size=31,
    )
    count = result.eligible_count_replicates
    expected = (2*count[:,0] - count[:,1]) / count.sum(axis=1)
    np.testing.assert_allclose(result.global_replicates[:,0], expected)
    np.testing.assert_allclose(result.global_replicates[:,1], 2*expected+7)
    np.testing.assert_allclose(result.global_replicates[:,2], -7)
    assert np.std(expected) > .05  # Would vanish if observed eligible mixture were fixed.
    assert np.std(count[:,0]) > 0 and np.std(count[:,1]) > 0
    assert result.summary["global_family_size"] == 3
    assert result.summary["effects"][2]["simultaneous_lower"] == pytest.approx(-7)
    assert result.summary["effects"][2]["simultaneous_upper"] == pytest.approx(-7)


def test_fixed_seed_determinism_and_joint_band_family():
    rng = np.random.default_rng(123)
    gains = rng.normal(size=(100, 2))
    ids = np.arange(100) % 2
    kwargs = dict(labels=["a","b"], n_resamples=150, seed=44, batch_size=17)
    first = stratified_paired_bootstrap(gains, ids, {0:50,1:50}, **kwargs)
    second = stratified_paired_bootstrap(gains, ids, {0:50,1:50}, **kwargs)
    np.testing.assert_array_equal(first.global_replicates, second.global_replicates)
    np.testing.assert_array_equal(first.eligible_count_replicates, 50)
    joint = first.summary["joint_primary_group_max_abs_t_critical"]
    assert all(joint >= value for value in first.summary["per_primary_group_max_abs_t_critical"].values())
    with pytest.raises(ValueError, match="launched >= eligible"):
        stratified_paired_bootstrap(gains, ids, {0:49,1:50}, **kwargs)
    with pytest.raises(TimeoutError):
        stratified_paired_bootstrap(gains, ids, {0:50,1:50}, deadline_unix=1, **kwargs)


def test_mean_seed_score_is_not_ensemble_and_information_sign_is_final():
    raw = np.zeros((40,5))
    corrected = np.broadcast_to(np.arange(5), (40,5))
    event, summary = paired_seed_gains(raw, corrected)
    np.testing.assert_array_equal(event, 2)
    assert summary["is_density_ensemble_log_score"] is False
    labels = primary_labels()
    contrasts = planned_contrasts()
    assert len(labels) == 8 and len(contrasts) == 16
    values = np.array([0 if x.startswith("Hidden5") else 2 if x.startswith("Full5") else 1 for x in labels])
    assert contrasts["gain_change:Full5_minus_Hidden5/f010"] @ values == 2
    assert contrasts["gain_change:Coarse2_minus_Hidden5/f100"] @ values == 1
    assert practical_effect(.0015)["geometric_mean_paired_density_ratio_percent_change"] == pytest.approx(.1501125562711)


def test_uniform_pit_and_separate_shape_diagnostics():
    u = (np.arange(10000) + .5) / 10000
    pit = pit_diagnostics(np.log(u), np.log1p(-u), np.zeros(len(u), dtype=int))["groups"][0]
    assert pit["pit_median"] == pytest.approx(.5)
    assert pit["median_coverage"] == pytest.approx(.5)
    assert pit["predictive_iqr_coverage"] == pytest.approx(.5)
    assert pit["predictive_central90_coverage"] == pytest.approx(.9)
    assert pit["lower_05_frequency"] == pytest.approx(.05)
    assert pit["upper_01_frequency"] == pytest.approx(.01)
    with pytest.raises(ValueError, match="inconsistent"):
        pit_diagnostics(np.log(u), np.log1p(-u)-.01, np.zeros(len(u), dtype=int))
    original = location_scale_shape(u, np.zeros(len(u), dtype=int), units="synthetic")["groups"][0]
    shifted = location_scale_shape(u+5, np.zeros(len(u), dtype=int), units="synthetic")["groups"][0]
    assert shifted["median"] - original["median"] == pytest.approx(5)
    assert shifted["iqr"] == pytest.approx(original["iqr"])
    assert shifted["quantile_asymmetry_10_50_90"] == pytest.approx(original["quantile_asymmetry_10_50_90"])


def test_source_56_partition_nested_burden_and_global_mu_invariance():
    fine = np.repeat(np.arange(56), 6)
    stratum = np.tile(np.arange(6), 56)
    p = .3 + (fine+.5)*.7/56
    t = np.array([.1,.3,1.])[stratum//2].astype(np.float32)
    q = np.where(stratum % 2, 1, -1)
    y = p + .3*t + .1*q
    first = source_location_burden(y, p, t, q, units="synthetic")
    shifted = source_location_burden(y+10, p, t, q, units="synthetic")
    a = np.array([x["event_weighted_position_variance"] for x in first["contexts"]])
    b = np.array([x["event_weighted_position_variance"] for x in shifted["contexts"]])
    np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-15)
    assert a[0] > a[1] > a[2] > 0
    assert first["Full5"]["exact_conditioning_unobserved_momentum_mean_variance"] == 0


def _fresh_metadata_fixture(tmp_path):
    from representation_flow.experiment_contract import save_json, sha256
    plan = tmp_path/'analysis_plan.json'
    save_json(plan, {"status":"FROZEN_FRESH_STATISTICAL_ANALYSIS"})
    models = []
    for i in range(240):
        models.append({"model_id":f"model{i}","cohort":"information80" if i<80 else "legacy160",
                       "root_key":"synthetic","checkpoint_kind":"information" if i<80 else "legacy",
                       "labels":{"seed":i},"experiment_identity":"synthetic", "artifacts":{
                           "checkpoint":{"sha256":f"{i:064x}"}}})
    union = tmp_path/'union.json'
    save_json(union,{"schema":"fresh-model-union/v1","status":"COMPLETE_FROZEN_FRESH_MODEL_UNION",
                     "expected_models":240,"models":models})
    roster = tmp_path/'roster.json'
    save_json(roster,{"schema":"fresh-model-roster/v1","expected_models":240,"models":models,
                      "expected_model_ids":[m['model_id'] for m in models]})
    fresh_plan = tmp_path/'fresh_plan.json'
    save_json(fresh_plan,{"schema":"fresh-catalog-protocol/v1","status":"FROZEN_FRESH_CATALOG",
                         "analysis_plan":{"sha256":sha256(plan)},"model_union":{"sha256":sha256(union)},
                         "source_files":[{"sha256":sha256(roster)}],
                         "catalogs":[{"block_id":b,"run_id":f"test-{b}"} for b in ('C','D')]})
    generation = tmp_path/'generation'
    cells=[{"group_id":g,"launched_events":12000,"eligible_events":11900,
            "launched":12000,"eligible":11900,"ineligible":100,"aborted":0,
            "statuses":{"forward":11900,"nonforward":100}} for g in range(42)]
    save_json(generation/'C/fresh_catalog_manifest.json',{
        "schema":"fresh-catalog-stratum-counts/v1","status":"COMPLETE_42_STRATUM_COUNTS",
        "technical_aborted":0,"run_id":"test-C","events_sha256":"f"*64,"cells":cells})
    save_json(generation/'GENERATION_COMPLETE.json',{
        "status":"COMPLETE_FRESH_GEANT4_CATALOGS_NO_MODEL_SCORING",
        "plan_sha256":sha256(fresh_plan),"analysis_sha256":sha256(plan),
        "catalogs":[{"block_id":b,"run_id":f"test-{b}",
                     "stratum_counts_sha256":sha256(generation/'C/fresh_catalog_manifest.json')} for b in ('C','D')]})
    score_models=[]
    for model in models:
        score_models.append({**{key:value for key,value in model.items() if key!='artifacts'},
                             "checkpoint_sha256":model['artifacts']['checkpoint']['sha256'],
                             "status":"COMPLETE_FROZEN_MODEL_SCORE"})
    manifest={"schema":"information-catalog-scores/v1","status":"COMPLETE_FROZEN_240_CATALOG_SCORES",
              "block_id":"C","plan_sha256":sha256(fresh_plan),"union_sha256":sha256(union),
              "generation_sha256":sha256(generation/'GENERATION_COMPLETE.json'),"models":score_models,
              "source_files":{},"failures":[],"events_sha256":"f"*64,
              "counts_sha256":sha256(generation/'C/fresh_catalog_manifest.json'),
              "launched_counts":{str(g):12000 for g in range(42)},
              "eligible_counts":{str(g):11900 for g in range(42)}}
    score_root=tmp_path/'scores/C'
    save_json(score_root/'score_manifest.json',manifest)
    save_json(score_root.parent/'D/score_manifest.json',dict(manifest,block_id='D'))
    parent={"status":"COMPLETE_FROZEN_240_CD_SCORES","failures":[],
            **{key:manifest[key] for key in ('plan_sha256','union_sha256','generation_sha256')},
            "roster_sha256":sha256(roster),
            "catalogs":[{"block_id":b,"path":f'{b}/score_manifest.json',"models":240,
                         "sha256":sha256(score_root.parent/b/'score_manifest.json')} for b in ('C','D')]}
    save_json(score_root.parent/'scoring_summary.json',parent)
    args=SimpleNamespace(plan=plan,union=union,roster=roster,fresh_plan=fresh_plan,
                         generation_root=generation,score_root=score_root)
    return args,manifest,parent


def test_fresh_analysis_binds_parent_plan_roster_and_checkpoint_metadata(tmp_path):
    from representation_flow.run_information_analysis import validate_fresh_provenance
    from representation_flow.experiment_contract import save_json
    args,manifest,parent=_fresh_metadata_fixture(tmp_path)
    assert validate_fresh_provenance(args,manifest)['counts_sha256']==manifest['counts_sha256']
    manifest['models'][0]['checkpoint_sha256']='f'*64
    with pytest.raises(ValueError,match='checkpoint'):
        validate_fresh_provenance(args,manifest)
    manifest['models'][0]['checkpoint_sha256']='0'*64
    parent['status']='RUNNING'
    save_json(args.score_root.parent/'scoring_summary.json',parent)
    with pytest.raises(ValueError,match='both predeclared'):
        validate_fresh_provenance(args,manifest)


def test_group_seed_variation_keeps_five_seeds_distinct_from_event_se(tmp_path):
    from representation_flow.run_information_analysis import summarize_component_seed_variation
    rows=[]
    for seed,delta in zip(range(3101,3106),[-2.,-1.,0.,1.,2.]):
        raw=_score(100)
        corrected={key:value.copy() for key,value in raw.items()}
        for key in ('base_logpdf','log_q_z','log_p_y'):
            corrected[key]+=delta
        report=paired_component_summary(raw,corrected,np.zeros(100,dtype=int))
        rows.append({'context':'Hidden5','fraction_tag':'f010','seed':seed,
                     'physical_groups':report,'momentum_groups':report})
    groups,components=summarize_component_seed_variation(rows)
    assert len(groups)==1 and len(components)==10
    assert groups[0]['sample_sd']==pytest.approx(np.sqrt(2.5))
    assert groups[0]['negative_seeds']==2 and groups[0]['positive_seeds']==2 and groups[0]['zero_seeds']==1
    assert [groups[0][f'seed_{seed}'] for seed in range(3101,3106)]==[-2.,-1.,0.,1.,2.]


def test_complete_new80_adapter_exports_all_pairs_and_seed_groups(tmp_path):
    from representation_flow.engine import RepresentationPreprocessor
    from representation_flow.run_information_analysis import new_model_diagnostics, write_common_outputs
    from representation_flow.information_uncertainty import CONTEXT_ORDER, FRACTION_ORDER
    gid=np.repeat(np.arange(42),2)
    common={'group_id':gid,'p_in':.3+(gid//6+.5)*.1,
            'thickness':np.asarray([.1,.3,1.])[gid%6//2],
            'charge':np.where(gid%2,1,-1),'y':np.linspace(.00001,.0001,len(gid))}
    models=[]
    saved={}
    for fraction in FRACTION_ORDER:
        for ci,context in enumerate(CONTEXT_ORDER):
            for si,seed in enumerate(range(3101,3106)):
                for representation in ('Raw','CorrectedPhysics'):
                    key=f'{fraction}-{context}-{seed}-{representation}'
                    pre=RepresentationPreprocessor(representation,np.zeros(1),0.,.0001)
                    models.append({'model_id':key,'labels':{'context_label':context,'fraction_tag':fraction,
                         'seed':seed,'capacity':'Small','representation':representation},'preprocessor':pre.to_dict()})
                    row=_score(len(gid))
                    if representation=='CorrectedPhysics':
                        delta=.001*(ci+1)+.0001*si
                        for field in ('base_logpdf','log_q_z','log_p_y'):
                            row[field]=row[field]+delta
                    u=np.linspace(.01,.99,len(gid))
                    row.update(log_cdf_y=np.log(u),log_sf_y=np.log1p(-u),base_value=np.zeros(len(gid)),
                               z=np.zeros(len(gid)),b=np.ones(len(gid)))
                    saved[key]=row
    gains,seed_rows,summaries,components,predictive,residual=new_model_diagnostics(
        models,lambda model:saved[model['model_id']],common,lambda:None)
    assert gains.shape==(84,8) and len(seed_rows)==80 and len(summaries)==16
    assert len(components)==40 and len(predictive)==80 and len(residual)==4
    endpoint=next(r for r in summaries if r['contrast']=='gain_change:Full5_minus_Hidden5/f010')
    assert endpoint['mean']==pytest.approx(.003)
    write_common_outputs(tmp_path,[{'nll':1.}],seed_rows,summaries,components,predictive,residual)
    import csv
    with (tmp_path/'group_seed_variation.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==336
    assert float(rows[0]['sample_sd'])==pytest.approx(np.std(np.arange(5)*.0001,ddof=1))
