"""Export the completed JVS evaluation and validate its portable result bundle.

Only publication copies are normalized. Original run manifests and resume state
remain untouched. The checks use saved CSVs and the Python standard library;
audio, inference models and the evaluation environments are not needed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'artifacts/quality/runs/20261019-jvs-one-latest-eacabcd'
DEFAULT_OUT = ROOT / 'benchmark/results/jvs-latest-20261019'
EXTERNAL_LIBRARY = re.compile(r'(?<=-DVulkan_LIBRARY=)/[^\s"\',]+')
ABSOLUTE_PATH = re.compile(r'(?<![\w.:/])/[A-Za-z][\w.-]*/|(?<!\w)[A-Za-z]:[\\/]')
MODES = {
    'official_bf16_common', 'cpp_f16_kaiser_common', 'cpp_f16_hann_common',
    'cpp_fp32_hann_common', 'official_fp32_common', 'cpp_fp32_hann_strict',
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def normalized(data):
    text = data.decode('utf-8').replace(str(ROOT) + '/', '')
    text = EXTERNAL_LIBRARY.sub('${Vulkan_LIBRARY}', text)
    if ABSOLUTE_PATH.search(text):
        raise ValueError('Unrecognized absolute path in publication input')
    return text.encode('utf-8')


def sources():
    pairs = {'evaluation/RESULTS.ja.md': RUN / 'report/LATEST.ja.md'}
    for name in ['summary.csv', 'per-seed.csv', 'comparisons.csv',
                 'latest-paired-differences.csv', 'coverage.json', 'failures.csv']:
        pairs['evaluation/' + name] = RUN / 'report' / name
    for name in ['manifest.json', 'build-manifest.json', 'validation.json', 'status.json']:
        pairs['evaluation/' + name] = RUN / name
    diagnostic = RUN / 'analysis/confusion'
    for name in [
        'RESULTS.ja.md', 'CONTROLLED.ja.md', 'analysis.json', 'manifest.json',
        'selection.json', 'hann-references.json', 'validation.json', 'status.json',
        'decisions.csv', 'summary.csv', 'per-speaker.csv', 'pairs-by-case.csv',
        'paired-outcomes.csv', 'paired-outcome-summary.csv', 'error-margins.csv',
        'speaker-gaps.csv', 'confusion-comparison.csv', 'concentration.csv',
        'natural-decisions.csv', 'controlled-decisions.csv', 'controlled-summary.csv',
        'controlled-per-speaker.csv', 'controlled-differences.csv',
    ]:
        pairs['confusion/' + name] = diagnostic / name
    return pairs


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f'Metric mismatch: {actual} != {expected}')


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def validate(directory, with_sources=False):
    provenance = read_json(directory / 'provenance.json')
    for name, record in provenance['files'].items():
        path = directory / name
        data = path.read_bytes()
        checked(sha256(data) == record['sha256'], 'Published hash mismatch: ' + name)
        checked(len(data) == record['bytes'], 'Published size mismatch: ' + name)
        checked(not ABSOLUTE_PATH.search(data.decode()), 'Absolute path remains: ' + name)
        if with_sources:
            original = (ROOT / record['source']).read_bytes()
            checked(sha256(original) == record['source_sha256'], 'Source changed: ' + name)
            checked(normalized(original) == data, 'Unexpected transformation: ' + name)
    for path in directory.rglob('*.json'):
        read_json(path)
    for path in directory.rglob('*'):
        if path.is_file():
            checked(not ABSOLUTE_PATH.search(path.read_text()), 'Absolute path remains: ' + str(path.relative_to(directory)))

    evaluation = directory / 'evaluation'
    checked(read_json(evaluation / 'status.json')['status'] == 'complete', 'Evaluation incomplete')
    coverage = read_json(evaluation / 'coverage.json')
    checked(coverage['generation'] == {'ok': 15000}, 'Unexpected evaluation coverage')
    checked({(r['scorer'], r['status'], r['count']) for r in coverage['scores']} ==
            {('campplus', 'ok', 15000), ('whisper', 'ok', 15000)}, 'Unexpected scorer coverage')
    checked((evaluation / 'failures.csv').read_bytes() == b'', 'Evaluation contains failures')

    keys = ['dataset', 'case', 'mode', 'kind', 'metric', 'success_only']
    seeds = defaultdict(list)
    for row in rows(evaluation / 'per-seed.csv'):
        seeds[tuple(row[k] for k in keys)].append(row)
    for row in rows(evaluation / 'summary.csv'):
        group = seeds[tuple(row[k] for k in keys)]
        checked({r['seed'] for r in group} == set('01234') and len(group) == 5, 'Seed coverage mismatch')
        values = [float(r['value']) for r in group]
        close(statistics.mean(values), float(row['mean']))
        close(statistics.pstdev(values), float(row['population_std']))

    diagnostic = directory / 'confusion'
    checked(read_json(diagnostic / 'status.json')['status'] == 'complete', 'Diagnostic incomplete')
    decisions = rows(diagnostic / 'decisions.csv')
    checked(len(decisions) == 15000 and len({r['id'] for r in decisions}) == 15000, 'Decision coverage mismatch')
    by_case = defaultdict(list)
    for row in decisions:
        checked((row['predicted'] == row['speaker']) == (row['correct'] == 'True'), 'Invalid speaker decision')
        by_case[row['case']].append(row)
    checked(len(by_case) == 6, 'Unexpected evaluation cases')
    for row in rows(diagnostic / 'summary.csv'):
        group = by_case[row['case']]
        checked(len(group) == 2500 and len({r['speaker'] for r in group}) == 100, 'Speaker coverage mismatch')
        close(statistics.mean(r['correct'] == 'True' for r in group), float(row['top1']))
        close(statistics.mean(float(r['true_similarity']) for r in group), float(row['true_similarity']))
        for metric, value in [('speaker_top1', row['top1']), ('camplus_cosine', row['true_similarity'])]:
            reported = next(r for r in rows(evaluation / 'summary.csv') if r['case'] == row['case']
                            and r['metric'] == metric and r['kind'] == 'one' and r['success_only'] == 'False')
            close(float(value), float(reported['mean']))
    counts = Counter((r['case'], r['speaker'], r['predicted']) for r in decisions if r['correct'] == 'False')
    checked(counts == Counter({(r['case'], r['speaker'], r['predicted']): int(r['count'])
                              for r in rows(diagnostic / 'pairs-by-case.csv')}), 'Confusion pair counts differ')
    for row in rows(diagnostic / 'confusion-comparison.csv'):
        for prefix, case in [('official', 'official-{}-bf16'), ('cpp', 'speech-{}-f16'), ('previous', 'previous-{}-f16')]:
            checked(counts[case.format(row['variant']), row['speaker'], row['predicted']] == int(row[prefix + '_count']),
                    'Confusion comparison differs from decisions')
    natural = rows(diagnostic / 'natural-decisions.csv')
    checked(Counter(r['role'] for r in natural) == {'reference': 100, 'natural_target': 500}, 'Natural recording coverage mismatch')
    checked(all(r['correct'] == 'True' for r in natural), 'Natural recording classification changed')

    controlled = rows(diagnostic / 'controlled-decisions.csv')
    selection = read_json(diagnostic / 'selection.json')
    checked(len(controlled) == 2400, 'Controlled generation count mismatch')
    groups = defaultdict(list)
    for row in controlled:
        checked(row['speaker'] in selection['speakers'], 'Unselected diagnostic speaker')
        checked(row['seed'] in set('01234'), 'Invalid diagnostic seed')
        groups[row['variant'], row['item'], row['seed']].append(row)
    checked(len(groups) == 400, 'Controlled group count mismatch')
    for group in groups.values():
        checked(len(group) == 6 and {r['mode'] for r in group} == MODES, 'Missing controlled mode')
        checked(len({(r['noise_sha256'], r['frames_expected']) for r in group}) == 1, 'Noise/duration not shared')
        official = next(r for r in group if r['mode'] == 'official_fp32_common')
        strict = next(r for r in group if r['mode'] == 'cpp_fp32_hann_strict')
        checked(official['predicted'] == strict['predicted'], 'Strict FP32 speaker predictions differ')
    strata = {**{k: set(selection[k]) for k in ['affected', 'controls']}, 'all': set(selection['speakers'])}
    for row in rows(diagnostic / 'controlled-summary.csv'):
        group = [r for r in controlled if r['variant'] == row['variant'] and r['mode'] == row['mode']
                 and r['speaker'] in strata[row['stratum']]]
        checked(len(group) == int(row['takes']), 'Controlled summary count mismatch')
        close(statistics.mean(r['correct'] == 'True' for r in group), float(row['top1']))
        close(statistics.mean(float(r['true_similarity']) for r in group), float(row['similarity']))
        close(statistics.mean(float(r['margin']) for r in group), float(row['margin']))
    checked(read_json(diagnostic / 'validation.json') == dict(generations=2400, scores=2400, groups=400,
            expected_modes=6, common_noise_and_frames=True, natural_scores=600), 'Original diagnostic validation changed')
    return dict(published_copies=len(provenance['files']), evaluation_decisions=len(decisions),
                controlled_decisions=len(controlled), controlled_groups=len(groups), natural_decisions=len(natural),
                five_seed_summaries_recomputed=True, confusion_counts_recomputed=True,
                controlled_summaries_recomputed=True, strict_fp32_predictions_identical=True,
                common_noise_and_frames=True, source_copies_verified=with_sources,
                published_hashes_verified=True, environment_specific_paths_found=0,
                original_artifacts_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--check', action='store_true', help='Validate published files without the original run')
    args = parser.parse_args()
    directory = args.out.resolve()
    if args.check:
        print(json.dumps(validate(directory), ensure_ascii=False, indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, source in sources().items():
        original = source.read_bytes()
        data = normalized(original)
        dest = directory / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        files[name] = dict(source=str(source.relative_to(ROOT)), source_sha256=sha256(original),
                           sha256=sha256(data), bytes=len(data))
        if original != data:
            files[name]['transformation'] = 'Repository paths made relative; external Vulkan loader path replaced by ${Vulkan_LIBRARY}; all other values unchanged'
    write_json(directory / 'provenance.json', dict(format=1, measured_on='2026-10-19', files=files,
        path_base='repository root', external_paths='${Vulkan_LIBRARY} denotes the runtime Vulkan loader; its installation directory is not published',
        code_note='Measured code hashes are retained in the original manifests. The published run.py discovers the Vulkan loader using scripts/native_toolchain.py instead of a fixed system path. Measurement/scoring outputs are unchanged.'))
    validation = validate(directory, with_sources=True)
    write_json(directory / 'validation.json', validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
