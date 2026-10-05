"""The shared TXT-to-histogram steps for the two dataset scripts."""

import ast
from collections import Counter
from pathlib import Path
import re
import tempfile

import numpy as np


LINE = re.compile(r'^\s*(\d+)\s+(\[[^\]]*\])\s+(\[[^\]]*\])\s+(\[[^\]]*\])\s+(\d+)\s*$')
FILE_PREFIX = 'weak_pnr_100000_'
MODES = 8
SHOTS = 100000
VACUUM = (0,) * MODES


def raw_files(folder):
    folder = Path(folder)
    paths = {}
    for path in folder.glob(FILE_PREFIX + '*.txt'):
        try:
            idx = int(path.stem[len(FILE_PREFIX):])
        except ValueError:
            continue
        paths[idx] = path
    if not paths:
        raise FileNotFoundError(f'No {FILE_PREFIX}*.txt files in {folder}')
    return paths


def read_file(path):
    rows = []
    eom_seen = set()
    total = 0
    with path.open(encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            if not line.strip():
                continue
            found = LINE.fullmatch(line.rstrip('\n'))
            if found is None:
                raise ValueError(f'{path.name}, line {line_num}: unexpected raw format')
            herald = ast.literal_eval(found[2])
            signal = ast.literal_eval(found[3])
            eom = ast.literal_eval(found[4])
            if (not isinstance(herald, list) or not isinstance(signal, list)
                    or len(herald) != MODES or len(signal) != MODES
                    or any(type(n) is not int or n < 0 for n in herald + signal)):
                raise ValueError(f'{path.name}, line {line_num}: invalid photon-number pattern')
            if not isinstance(eom, list) or len(eom) != 1 or eom[0] not in (0, 1):
                raise ValueError(f'{path.name}, line {line_num}: invalid EOM value')
            count = int(found[5])
            eom_seen.add(eom[0])
            total += count
            rows.append((tuple(herald), tuple(signal), count))
    if not rows or len(eom_seen) != 1 or total != SHOTS:
        raise ValueError(f'{path.name}: rows={len(rows)}, EOM={eom_seen}, shots={total}; expected one EOM and {SHOTS} shots')
    return eom_seen.pop(), rows


def audit(paths):
    good = {0: [], 1: []}
    bad = Counter()
    for num, idx in enumerate(sorted(paths), 1):
        try:
            eom, _ = read_file(paths[idx])
        except (OSError, ValueError, SyntaxError) as exc:
            bad[str(exc).split(':', 1)[-1].strip().split(';')[0]] += 1
        else:
            good[eom].append(idx)
        if num % 500 == 0:
            print(f'Checked {num}/{len(paths)} raw files', flush=True)
    print(f'Usable files: GBS={len(good[0])}, SBS/TBS={len(good[1])}; skipped={sum(bad.values())}')
    if bad:
        print('Skipped file reasons:', dict(bad))
    return good


def periods(indices, kind):
    if not indices:
        return []
    chunks = [[indices[0]]]
    for idx in indices[1:]:
        if idx - chunks[-1][-1] > 5:
            chunks.append([])
        chunks[-1].append(idx)

    out = []
    for files in chunks:
        # Twenty 0.1 s files on each side of a switch are left out.
        if files[-1] - files[0] + 1 - 40 >= 10:
            out.append({'kind': kind, 'files': files, 'first': files[0], 'last': files[-1]})
    return out


def make_windows(good):
    gbs = periods(good[0], 'gbs')
    other = periods(good[1], 'sbs_tbs')
    if not gbs or not other:
        raise ValueError('Need complete GBS and SBS/TBS acquisition periods')

    start, end = gbs[0]['first'], other[-1]['last']
    all_periods = [p for p in gbs + other if start <= p['first'] and p['last'] <= end]
    all_periods.sort(key=lambda p: p['first'])
    windows = {'gbs': [], 'sbs': [], 'tbs': []}

    for period_id, p in enumerate(all_periods):
        kept = [i for i in p['files'] if p['first'] + 20 <= i <= p['last'] - 20]
        for pos in range(0, len(kept) - 9, 15):
            files = kept[pos:pos+10]
            if max(b - a for a, b in zip(files[:-1], files[1:])) > 5:
                continue
            kind = p['kind']
            row = {'files': files, 'period': period_id, 'window': pos // 15}
            if kind == 'gbs':
                windows['gbs'].append(row)
            else:
                # TBS and SBS use the same EOM=1 shots, but SBS is conditioned
                # on the measured herald while TBS keeps all signal shots.
                windows['tbs'].append(row)
                windows['sbs'].append(row)

    # The old panel gives the three families the same number of global bins.
    n = min(len(windows[k]) for k in windows)
    if n < 2:
        raise ValueError(f'Too few complete one-second windows: { {k: len(v) for k,v in windows.items()} }')
    for kind in windows:
        windows[kind] = windows[kind][:n]
        for i, row in enumerate(windows[kind]):
            row['bin'] = i
    print(f'Complete 1 s bins per family: {n} (10 files read, 5 files skipped)')
    return windows


def select_bins(windows, method, value):
    splits = {'train': {}, 'test': {}}
    for kind, rows in windows.items():
        if method == 'interpol':
            train = [row for i, row in enumerate(rows) if i % (value + 1) == 0]
            test = [row for i, row in enumerate(rows) if i % (value + 1) != 0]
        else:
            cutoff = int(len(rows) * value / 100)
            if not 0 < cutoff < len(rows):
                raise ValueError(f'{kind}: {value}% gives an empty train or test split')
            train, test = rows[:cutoff], rows[cutoff:]
        splits['train'][kind], splits['test'][kind] = train, test
        print(f'{kind.upper()}: {len(train)} train bins, {len(test)} test bins')
    return splits


def build_states(splits, paths, run):
    sets = {'train': [], 'test': []}
    for part in ('train', 'test'):
        for kind, windows in splits[part].items():
            for row in windows:
                signal_hist = Counter()
                by_herald = {}
                for idx in row['files']:
                    eom, raw_rows = read_file(paths[idx])
                    if eom != (0 if kind == 'gbs' else 1):
                        raise ValueError(f'{paths[idx]} changed EOM since the file audit')
                    for herald, signal, count in raw_rows:
                        signal_hist[signal] += count
                        # Exclude the *input* vacuum herald for SBS. An output
                        # vacuum event is still a real detector outcome and
                        # stays in the conditional photon-number histogram.
                        if kind == 'sbs' and herald != VACUUM:
                            by_herald.setdefault(herald, Counter())[signal] += count

                if kind != 'sbs':
                    add_state(sets[part], run, kind, row, signal_hist)
                else:
                    for herald, hist in sorted(by_herald.items()):
                        if sum(hist.values()) >= 100:
                            add_state(sets[part], run, kind, row, hist, herald)
    if not sets['train'] or not sets['test']:
        raise ValueError('No states in train or test after SBS herald selection')
    for part in sets:
        counts = Counter(s['kind'] for s in sets[part])
        print(f'{part}: {dict(counts)} states before oversampling')
        if set(s['label'] for s in sets[part]) != {0, 1}:
            raise ValueError(f'{part}: both classical and nonclassical states are needed')
    train_ids = {s['id'] for s in sets['train']}
    test_ids = {s['id'] for s in sets['test']}
    if train_ids & test_ids:
        raise ValueError(f'Train/test state overlap: {next(iter(train_ids & test_ids))}')
    return sets


def add_state(target, run, kind, row, hist, herald=None):
    if not hist or sum(hist.values()) <= 0:
        return
    if herald is None:
        name = f'{kind}_bin_{row["bin"]}'
        herald_text = ''
    else:
        name = f'sbs_bin_{row["bin"]}_h_' + '_'.join(map(str, herald))
        herald_text = '-'.join(map(str, herald))
    # Labels describe the prepared family: thermal is classical (0), while
    # the GBS and non-vacuum-heralded SBS inputs are labelled nonclassical (1).
    target.append({'id': f'{run}_{name}', 'kind': kind, 'label': int(kind != 'tbs'),
                   'herald': herald_text, 'hist': hist, 'period': row['period'],
                   'bin': row['bin'], 'window': row['window'],
                   'start': row['files'][0], 'end': row['files'][-1], 'run': run})


def grow(items, target, stage):
    if not items and target:
        raise ValueError(f'No states to oversample in {stage}')
    result = list(items)
    for i in range(target - len(items)):
        source = items[i % len(items)]
        copy = source.copy()
        copy['id'] = source['id'] + f'__oversample_{stage}_{i+1:06d}'
        result.append(copy)
    return result


def oversample(states):
    groups = {kind: [s for s in states if s['kind'] == kind] for kind in ('tbs', 'gbs', 'sbs')}
    if any(not group for group in groups.values()):
        raise ValueError('Train needs TBS, GBS and SBS states for family balancing')
    goal = max(len(groups['gbs']), len(groups['sbs']))
    groups['gbs'] = grow(groups['gbs'], goal, 'gbs')
    groups['sbs'] = grow(groups['sbs'], goal, 'sbs')
    classical = groups['tbs']
    nonclassical = groups['gbs'] + groups['sbs']
    goal = max(len(classical), len(nonclassical))
    return grow(classical, goal, 'classical') + grow(nonclassical, goal, 'nonclassical')


def write_h5(path, states, method, value):
    import h5py

    n = len(states)
    width = max(len(s['hist']) for s in states)
    images = np.zeros((n, MODES, width), dtype=np.float32)
    occ = np.zeros((n, width), dtype=np.float32)
    for i, state in enumerate(states):
        for j, (pattern, count) in enumerate(sorted(state['hist'].items())):
            images[i, :, j] = pattern
            occ[i, j] = count

    strings = h5py.string_dtype('utf-8')
    with h5py.File(path, 'w') as out:
        out.create_dataset('images', data=images, compression='gzip')
        out.create_dataset('occ', data=occ, compression='gzip')
        out.create_dataset('labels', data=np.array([s['label'] for s in states], dtype=np.int8))
        meta = out.create_group('meta')
        for key, field in [('state_id', 'id'), ('kind', 'kind'), ('source_id', 'run'),
                           ('heralding_pattern', 'herald')]:
            meta.create_dataset(key, data=[s[field] for s in states], dtype=strings)
        for key, field in [('period_id', 'period'), ('global_bin', 'bin'),
                           ('window_id', 'window'), ('start_file_index', 'start'),
                           ('end_file_index', 'end')]:
            meta.create_dataset(key, data=np.array([s[field] for s in states], dtype=np.int32))
        out.attrs['split'] = method
        out.attrs['gap' if method == 'interpol' else 'train_percent'] = value
        out.attrs['input_vacuum_sbs'] = False
        out.attrs['output_vacuum_retained'] = True
    print(f'Wrote {path.name}: {images.shape}, labels {dict(Counter(s["label"] for s in states))}')


def build(raw_dir, out_root, run, method, values):
    paths = raw_files(raw_dir)
    windows = make_windows(audit(paths))
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for value in values:
        if method == 'interpol':
            name = f'{run}_interpol_gap{value}_allbins_oversampled_allEV'
        else:
            name = (f'{run}_extrapol_train{value}_test{100-value}_oversampled_'
                    'allSBSge100_noInputVac_keepOutputVac_allEV')
        target = out_root / name
        if target.exists():
            raise FileExistsError(f'Dataset already exists: {target}')

        bins = select_bins(windows, method, value)
        states = build_states(bins, paths, run)
        states['train'] = oversample(states['train'])

        # A failed run must not leave a half-built pair for the classifier.
        with tempfile.TemporaryDirectory(dir=out_root, prefix='.building_') as tmp:
            tmp = Path(tmp)
            write_h5(tmp / f'{name}_train.h5', states['train'], method, value)
            write_h5(tmp / f'{name}_test.h5', states['test'], method, value)
            tmp.rename(target)
        print('Dataset:', target)
