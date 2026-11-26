"""Presentation-only tables from complete saved C/D aggregates; no inference."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "manuscript/representation_v2/overleaf_submission"
FRESH = ROOT / "reports/information_extension_20261108/fresh_analysis"
SEEDS = ROOT / "reports/positioning_revision_20261108/seed_sensitivity/results"
OUT = None
AUDIT = None
INPUTS: dict[str, str] = {}
NUMERIC_CHECKS: list[dict] = []
OUTPUTS: list[Path] = []
KEYS = ("scope", "fraction_tag", "capacity", "context", "contrast")
CONTEXT = {"Hidden5": "N", "Coarse2": "M_2", "Coarse8": "M_8", "Full5": "E"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    INPUTS[str(path)] = sha(path)
    return path.read_text(encoding="utf-8-sig")


def rows(path):
    return list(csv.DictReader(read(path).splitlines()))


def put(name, value):
    path = OUT / name
    path.write_text(value, encoding="utf-8", newline="\n")
    OUTPUTS.append(path)
    return path


def same(a, b, where, atol=2e-13):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=atol):
        raise ValueError(f"numeric mismatch {where}: {a} != {b}")


def validate_summary(summary, seed_rows, fields):
    seen = set()
    for row in summary:
        key = tuple(row[x] for x in fields)
        assert key not in seen
        seen.add(key)
        matched = [s for s in seed_rows if tuple(s[x] for x in fields) == key]
        assert len(matched) == 5 and {int(s['seed']) for s in matched} == set(range(3101, 3106))
        values = [float(s['gain']) for s in matched]
        for name, expected in [('mean', statistics.mean(values)), ('sample_sd', statistics.stdev(values)),
                               ('min', min(values)), ('max', max(values))]:
            same(row[name], expected, (key, name))
        for name, expected in [('positive_seeds', sum(x > 0 for x in values)),
                               ('negative_seeds', sum(x < 0 for x in values)),
                               ('zero_seeds', sum(x == 0 for x in values))]:
            assert int(row[name]) == expected
    assert len(seed_rows) == len(summary) * 5


def raw_tex_rows(text):
    return [line for line in text.splitlines() if ' & ' in line and line.rstrip().endswith(r'\\')
            and not line.startswith(('Contrast', r'\toprule'))]


def verify_display(old, new):
    # Ignore the relabeled first column and compare every numerical cell verbatim.
    oldrows, newrows = raw_tex_rows(old), raw_tex_rows(new)
    assert len(oldrows) == len(newrows)
    for a, b in zip(oldrows, newrows):
        assert a.split(' & ')[1:] == b.split(' & ')[1:]
    return len(oldrows)


def secondary_table(block, summary, historic=False):
    selected = [r for r in summary if (r['scope'] == 'core60') == historic]
    assert len(selected) == (18 if historic else 23)
    suffix = 'historical' if historic else 'current'
    if historic:
        caption = (f"Earlier-model controls evaluated on fresh catalog {block}. "
                   r"$H$ is the erroneous preliminary center, retained only as a historical control. "
                   r"$R$ is Raw, $B$ is BB-centered, and $D(A,B)=\mathrm{NLL}(A)-\mathrm{NLL}(B)$. "
                   r"$N_{\mathrm{o}}$ is the earlier two-input conditioner (thickness and charge), distinct from the new five-input no-direct-momentum model; $E_{\mathrm{o}}$ has exact momentum. "
                   r"$E_{\mathrm{o}}-N_{\mathrm{o}}$ rows subtract the corresponding NLL contrasts. ")
        columns = r"Contrast & Inputs & Train (\%) & Mean (SD) & Seed range & $+/-/0$"
    else:
        caption = (f"Complete secondary center, capacity and training-fraction contrasts on fresh catalog {block}, "
                   r"all with exact momentum inputs. $R$, $B$, $L$ and $P$ denote Raw, BB-centered, Linear center and Polynomial-ridge center. "
                   r"$D(A,B)=\mathrm{NLL}(A)-\mathrm{NLL}(B)$, so positive values favor the second model (and negative $D(L,R)$ indicates lower mean NLL for $L$ than $R$). "
                   r"Small and Large have widths 32 and 128. "
                   r"$\Delta=D(R,B)$; the final five rows compare this gain across capacities or fractions, not absolute NLL. ")
        columns = r"Contrast & Capacity & Train (\%) & Mean (SD) & Seed range & $+/-/0$"
    caption += (r"Effects are in units of 0.001 nat per eligible event. Means, sample SDs, ranges and positive/negative/zero counts use five paired optimizer seeds. "
                r"These descriptive seed spreads are not event-bootstrap intervals. Catalogs C and D are reported separately.")
    lines = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{3pt}', r'\begin{longtable}{lllrrr}',
             r'\caption{' + caption + r'}\label{tab:editorial-secondary-' + suffix + '-' + block + r'}\\',
             r'\toprule ' + columns + r'\\ \midrule \endfirsthead',
             r'\toprule ' + columns + r'\\ \midrule \endhead']
    contrast_names = {'Raw_minus_CorrectedPhysics': r'$D(R,B)$',
                      'Learned_minus_CorrectedPhysics': r'$D(P,B)$',
                      'Simple_minus_Raw': r'$D(L,R)$',
                      'Raw_minus_HistoricalIncorrectPhysics': r'$D(R,H)$',
                      'HistoricalIncorrectPhysics_minus_CorrectedPhysics': r'$D(H,B)$'}
    last_group = None
    rendered = []
    for row in selected:
        f = row['fraction_tag']
        fraction = {'f010':'10', 'f030':'30', 'f100':'100', 'f010_minus_f100':'10--100'}[f]
        if historic:
            label = contrast_names[row['contrast']]
            middle = {'Full':r'$E_{\mathrm{o}}$', 'Hidden':r'$N_{\mathrm{o}}$', 'Full_minus_Hidden':r'$E_{\mathrm{o}}-N_{\mathrm{o}}$'}[row['context']]
            group = f
        else:
            label = contrast_names[row['contrast']]
            middle = row['capacity']
            group = (f, row['capacity'])
            if middle == 'Small_minus_Large':
                label, middle = r'$\Delta_{32}-\Delta_{128}$', 'Small--Large'
                group = 'capacity_gain'
            elif f == 'f010_minus_f100':
                label = r'$\Delta_{10}-\Delta_{100}$'
                group = 'fraction_gain'
        if last_group is not None and last_group != group:
            lines.append(r'\midrule')
        last_group = group
        numbers = [f"{1000*float(row['mean']):+.3f} ({1000*float(row['sample_sd']):.3f})",
                   f"[{1000*float(row['min']):+.3f}, {1000*float(row['max']):+.3f}]",
                   f"{row['positive_seeds']}/{row['negative_seeds']}/{row['zero_seeds']}"]
        line = ' & '.join([label, middle, fraction] + numbers) + r'\\'
        lines.append(line)
        rendered.append({'catalog': block, 'source_key': {k: row[k] for k in KEYS},
                         'file': f'{suffix}_secondary_{block}.tex', 'numeric_cells': numbers})
    lines += [r'\bottomrule', r'\end{longtable}', r'\endgroup', '']
    result = '\n'.join(lines)
    assert len(raw_tex_rows(result)) == len(selected)
    for row in rendered:
        assert ' & '.join(row['numeric_cells']) + r'\\' in result
    put(f'{suffix}_secondary_{block}.tex', result)
    NUMERIC_CHECKS.extend(rendered)


def relabel_primary(block, summary, uncertainty):
    path = PAPER / f'information_tables/fresh/fresh_{block}_primary16.tex'
    old = read(path)
    new = old.replace(r'\small', r'\normalsize')
    for internal, display in CONTEXT.items():
        new = new.replace(r'\mathrm{' + internal + '}', r'\mathrm{' + display + '}')
    new = new.replace('Raw-minus-Corrected NLL', 'Raw-minus-BB-centered NLL')
    new = new.replace('All quantities are in nats per eligible event.',
                      'All quantities are in nats per eligible event. ' +
                      r'$N,M_2,M_8,E$ denote No direct momentum, 2-bin momentum, 8-bin momentum and Exact momentum.')
    eligible = {'C':'499,285', 'D':'499,275'}[block]
    caption = (f'All 16 prespecified contrasts on catalog {block} ({eligible} eligible events). '
               r'$\Delta$ is mean Raw-minus-BB-centered NLL across five paired optimizer seeds; differences are more minus less information. '
               r'Intervals are approximate 95\% simultaneous event-bootstrap intervals within this catalog (2000 resamples), conditional on fixed models. '
               r'Seed SD describes optimizer variation, not interval uncertainty; $n_+$ counts positive seeds. Units: nat per eligible event.')
    new = re.sub(r'\\caption\{.*?\}\n(?=\\label)', lambda _: r'\caption{'+caption+'}\n', new, flags=re.S)
    n = verify_display(old, new)
    assert n == 16
    d = {r['name']:r for r in uncertainty}
    for line, row in zip(raw_tex_rows(new), summary):
        u = d[row['contrast']]
        same(row['mean'], u['estimate'], (block, row['contrast'], 'bootstrap estimator'))
        expected = [f"${float(row['mean']):+.6f}$", f"${float(row['sample_sd']):.6f}$",
                    f"${float(u['simultaneous_lower']):+.6f}$", f"${float(u['simultaneous_upper']):+.6f}$",
                    f"{row['positive_seeds']}/5"]
        assert line.split(' & ')[2:] == expected[:-1] + [expected[-1] + r' \\']
    put(f'fresh_{block}_primary16.tex', new)


def relabel_seeds(all_primary):
    saved = rows(SEEDS/'seed_sensitivity.csv')
    trows = rows(SEEDS/'exploratory_t_intervals.csv')
    assert len(saved) == len(trows) == 32
    bykey = {(r['catalog'], r['contrast']):r for r in saved}
    tkey = {(r['catalog'], r['contrast']):r for r in trows}
    for block, (summary, seeds) in all_primary.items():
        for row in summary:
            sr = bykey[block, row['contrast']]
            vals = [float(s['gain']) for s in seeds if s['contrast'] == row['contrast']]
            for k in ('mean', 'sample_sd', 'positive_seeds', 'negative_seeds', 'zero_seeds'):
                same(sr[k], row[k], (block, row['contrast'], k))
            same(sr['minimum'], min(vals), 'seed min')
            same(sr['maximum'], max(vals), 'seed max')
            loo = [(sum(vals)-v)/4 for v in vals]
            same(sr['leave_one_out_minimum_mean'], min(loo), 'LOO min')
            same(sr['leave_one_out_maximum_mean'], max(loo), 'LOO max')
            tr = tkey[block, row['contrast']]
            radius = float(tr['critical_t'])*statistics.stdev(vals)/math.sqrt(5)
            same(tr['lower'], statistics.mean(vals)-radius, 'exploratory t lower')
            same(tr['upper'], statistics.mean(vals)+radius, 'exploratory t upper')
    mapped_caption = ('$N,M_2,M_8,E$ denote No direct momentum, 2-bin momentum, 8-bin momentum and Exact momentum. '
                      r'Here $G$ is Raw-minus-BB-centered NLL.')
    counts = {}
    for name, expected_count in [('seed_adjacent_summary.tex',12), ('seed_full_summary.tex',32), ('seed_exploratory_t_summary.tex',32)]:
        old = read(PAPER/'seed_sensitivity'/name)
        new = old.replace(r'\small', r'\normalsize').replace('$H,C2,C8,F$ denote Hidden5, Coarse2, Coarse8 and Full5.', mapped_caption)
        if name == 'seed_exploratory_t_summary.tex':
            new = new.replace('Values are in $10^{-3}$ nat per eligible event.',
                              'Values are in $10^{-3}$ nat per eligible event. ' + mapped_caption)
        new = new.replace('All16 contrasts', 'All 16 contrasts')
        for internal, display in [('C2','M_2'),('C8','M_8'),('H','N'),('F','E')]:
            new = new.replace('G_{'+internal+'}', 'G('+display+')')
        # Plain adjacent-step cells have no math subscripts.
        for oldlabel, newlabel in [('C2--H',r'$M_2-N$'),('C8--C2',r'$M_8-M_2$'),('F--C8',r'$E-M_8$')]:
            new = new.replace(oldlabel, newlabel)
        oldrows, newrows = raw_tex_rows(old), raw_tex_rows(new)
        assert len(oldrows) == len(newrows) == expected_count
        for a, b in zip(oldrows, newrows):
            start = 2 if name == 'seed_adjacent_summary.tex' else 1
            assert a.split(' & ')[start:] == b.split(' & ')[start:]
        # Match every emitted numeric row against the independently checked saved CSV.
        if name != 'seed_adjacent_summary.tex':
            reference = saved if name == 'seed_full_summary.tex' else trows
            for line, row in zip(newrows, reference):
                cells = line.split(' & ')
                if name == 'seed_full_summary.tex':
                    expected = [f"{1000*float(row['mean']):.3f} ({1000*float(row['sample_sd']):.3f})",
                                f"[{1000*float(row['minimum']):.3f}, {1000*float(row['maximum']):.3f}]",
                                f"{row['positive_seeds']}/{row['negative_seeds']}/{row['zero_seeds']}",
                                f"[{1000*float(row['leave_one_out_minimum_mean']):.3f}, {1000*float(row['leave_one_out_maximum_mean']):.3f}]",
                                row['leave_one_out_direction']+r'\\']
                else:
                    expected = [f"{1000*float(row['mean']):.3f}",
                                f"[{1000*float(row['lower']):.3f}, {1000*float(row['upper']):.3f}]", row['direction'].replace('_',' ')+r'\\']
                assert cells[2:] == expected, (name, cells, expected)
        if name == 'seed_exploratory_t_summary.tex':
            new = new.replace('Optional exploratory Student-$t$ intervals on catalog ', 'Exploratory optimizer-seed intervals on catalog ').replace('; not the default sensitivity table.', '.')
        put(name, new)
        counts[name] = expected_count
    return counts


def main():
    import argparse
    global OUT, AUDIT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();OUT=args.output.resolve();AUDIT=OUT
    if OUT.exists():raise FileExistsError('choose a new presentation output')
    OUT.mkdir(parents=True)
    extension = ROOT/'reports/corrected_formal_extension_20261107/study'
    freeze = json.loads(read(extension/'FROZEN_CHECKPOINT_MANIFEST.json'))
    large = [m for m in freeze['models'] if m['capacity'] == 'Large']
    assert len(large) == 60
    for model in large:
        item = model['artifacts']['run_manifest']
        path = extension/item['path']
        assert sha(path) == item['sha256']
        run = json.loads(read(path))
        assert run['capacity'] == 'Large' and run['hidden_width'] == 128
    all_primary = {}
    for block in ('C','D'):
        report = json.loads(read(FRESH/block/'analysis.json'))
        assert report['status'] == 'COMPLETE_INFORMATION_FRESH_ANALYSIS'
        legacy = rows(FRESH/block/'legacy_contrast_summary.csv')
        legacy_seeds = rows(FRESH/block/'legacy_seed_contrasts.csv')
        assert len(legacy) == 41
        validate_summary(legacy, legacy_seeds, KEYS)
        secondary_table(block, legacy)
        secondary_table(block, legacy, historic=True)
        primary = rows(FRESH/block/'primary_contrast_summary.csv')
        primary_seeds = rows(FRESH/block/'primary_seed_contrasts.csv')
        assert len(primary) == 16
        validate_summary(primary, primary_seeds, ('contrast',))
        relabel_primary(block, primary, rows(FRESH/block/'primary_uncertainty.csv'))
        all_primary[block] = primary, primary_seeds
    seed_counts = relabel_seeds(all_primary)
    for name, names in [
        ('fresh_information_primary16.tex',['fresh_C_primary16.tex','fresh_D_primary16.tex']),
        ('current_secondary_CD.tex',['current_secondary_C.tex','current_secondary_D.tex']),
        ('historical_secondary_CD.tex',['historical_secondary_C.tex','historical_secondary_D.tex']),
        ('complete_secondary_CD.tex',['current_secondary_C.tex','current_secondary_D.tex','historical_secondary_C.tex','historical_secondary_D.tex'])]:
        put(name, '\n'.join(r'\input{editorial_tables/'+x+'}' for x in names)+'\n')
    put('display_names.tex', r'''\begin{table}[!htbp]
\centering\small
\caption{Paper display names and frozen implementation identifiers. Display names do not change saved models or protocol definitions.}
\label{tab:editorial-display-names}
\begin{tabular}{lll}
\toprule Paper display name & Table symbol & Frozen identifier\\
\midrule
No direct momentum & $N$ & Hidden5\\
2-bin momentum & $M_2$ & Coarse2\\
8-bin momentum & $M_8$ & Coarse8\\
Exact momentum & $E$ & Full5\\
Raw & $R$ & Raw\\
BB-centered & $B$ & CorrectedPhysics\\
Linear center & $L$ & Simple\\
Polynomial-ridge center & $P$ & Learned\\
\bottomrule
\end{tabular}
\end{table}
''')
    for name, before in INPUTS.items():
        assert sha(Path(name)) == before, f'Input modified: {name}'
    manifest = {'schema':'editorial-evidence-tables/v1','status':'NUMERIC_VERIFICATION_PASS_LAYOUT_PENDING',
                'primary_rows':32,'secondary_rows':82,'secondary_current_rows':46,'secondary_historical_rows':36,
                'seed_table_rows':seed_counts,'secondary_seed_rows_reconstructed':410,'primary_seed_rows_reconstructed':160,
                'all_secondary_contrasts_retained':True,'no_scientific_values_changed':True,
                'large_width_check':{'frozen_run_manifests_checked':60,'all_hashes_match_freeze':True,'hidden_width':128},
                'source_files':[{'path':p,'sha256':s} for p,s in INPUTS.items()],
                'outputs':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in OUTPUTS],
                'script':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},
                'secondary_numeric_rows':NUMERIC_CHECKS,
                'semantics':{'secondary':'Descriptive mean/sampleSD/min/max/sign counts across five paired optimizer seeds; no event CI was generated.',
                             'primary':'Original 16-family approximate95% simultaneous event-bootstrap intervals, fixed fitted models, C/D separate.',
                             'seed':'Post-hoc seed sensitivity. LOO is not a CI. Optional n=5 t intervals are nominal, unadjusted and separate from event bootstrap.',
                             'historical':'Earlier two-input Hidden is not the new nominal-five-input Hidden5. H is an erroneous-center control.',
                             'repeated_models':'New/earlier Exact-momentum Small10/100 states overlap; duplicated contrasts are not independent evidence.'}}
    (AUDIT/'TABLE_NUMERIC_VERIFICATION.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('status','primary_rows','secondary_rows','seed_table_rows')}))


if __name__ == '__main__':
    main()
