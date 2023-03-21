import argparse
from functools import reduce
import os
from os.path import join
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as hyperparams
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from datetime import datetime

PASTKEY = 'past'
FUTUREKEY = 'future'
TRAINKEY = 'train'
VALKEY = 'val'
LR = 'lr'
BS = 'bs'


def _process_field(event):
    name = event[0].name
    val = event[1]
    if isinstance(val, int) or isinstance(val, str) or isinstance(val, float):
        return {name: val}
    elif isinstance(val, RepeatedCompositeFieldContainer):
        return {
            i.tag: i.simple_value for i in val
        }
    else:
        fields = val.ListFields()
        fields = [_process_field(f) for f in fields]
        fields = reduce(lambda x, y: x | y, fields)
        return fields


def process_fields(fields):
    fields = [_process_field(f) for f in fields]
    return reduce(lambda x, y: x | y, fields)


def read_events(reader):
    raw_events = list(reader)
    return [process_fields(e.ListFields()) for e in raw_events]


def track_metric(events, key, metric, writer, best_key=None):
    best = float('inf')
    for e in events:
        if key in e.keys():
            writer.add_scalar(
                metric, e[key], global_step=e['step'], walltime=e['wall_time'])
            best = min(best, e[key])
    if best_key is not None:
        return {best_key: best}
    return {}


def get_all_metrics(events):
    metrics = set()
    for e in events:
        for k in e.keys():
            metrics.add(k)
    metrics.remove('step')
    metrics.remove('wall_time')
    metrics.remove('file_version')
    return list(metrics)


def is_empty(events):
    if len(events) > 1:
        return False
    if len(events) == 0:
        return True
    fields = events[0]
    if len(fields) != 2:
        return False
    for i, field in enumerate(list(fields.keys())):
        if field != 'wall_time' or field != 'file_version':
            return False
    return True


def meets_min_events(events):
    return len(events) > 50


def run_name_to_hparams(run_name):
    name = run_name.split('_')
    keys = [PASTKEY, FUTUREKEY, TRAINKEY, VALKEY, LR, BS]
    parsed = [name[i][len(keys[i]):] for i in range(len(keys))]
    n_past = int(parsed[0])
    n_future = int(parsed[1])
    n_train = int(parsed[2])
    n_val = int(parsed[3])
    lr = float(parsed[4])
    bs = int(parsed[5])
    tailoring = "_".join(name[6:])
    if tailoring == 'notailor_pde_emb':
        tailoring = 'None'
    elif tailoring == 'pde_emb' or tailoring == 'tailor_pde_emb':
        tailoring = 'PDE'
    elif tailoring == 'conv_emb' or tailoring == 'tailor_conv_emb':
        tailoring = 'Conv'
    return {
        'n_past': n_past,
        'n_future': n_future,
        'n_train': n_train,
        'n_val': n_val,
        'lr': lr,
        'bs': bs,
        'tailoring': tailoring,
    }


def process_run(run_dir, out_dir):
    run_name = run_dir.split('/')[-1]
    files = os.listdir(run_dir)
    files = list(filter(lambda fp: not (fp[-len('.pth'):] == '.pth'), files))
    nonempty = []
    for f in files:
        # top level events
        _f = join(run_dir, f)
        if os.path.isfile(_f):
            events = read_events(summary_iterator(_f))
            if not is_empty(events) and meets_min_events(events):
                nonempty.append((f, events))
    if len(nonempty) == 0:
        print(f'Run {run_dir} is empty. Skipping.')
        return
    assert len(
        nonempty) == 1, f'Found more than 1 high level run! {len(nonempty)}'
    top_f, top_events = nonempty[0]
    wall_time = top_events[0]['wall_time']
    run_time = datetime.fromtimestamp(wall_time).ctime()[
        4:].replace(' ', '-').replace(':', '.')
    hparams = run_name_to_hparams(run_name)
    run_name = f"{run_time}_past={hparams['n_past']}_future={hparams['n_future']}_tailor={hparams['tailoring']}"
    writer = SummaryWriter(log_dir=join(out_dir, run_name))
    tailoring = False
    if hparams['tailoring'] != 'None':
        tailoring = True
        max_tailor_steps = 2
        custom_scalars = {
            "Inner Loss": {
                "Train": ["Multiline", [f"Inner Loss/train/{i} Steps" for i in range(max_tailor_steps)]],
                "Val": ["Multiline", [f"Inner Loss/val/{i} Steps" for i in range(max_tailor_steps)]],
            },
            "SVG Loss": {
                "Train": ["Multiline", [f"SVG Loss/train/{i} Steps" for i in range(max_tailor_steps)]],
                "Val": ["Multiline", [f"SVG Loss/val/{i} Steps" for i in range(max_tailor_steps)]],
            },
        }
        writer.add_custom_scalars(custom_scalars)
    final_metrics = []
    # Add events from top log
    metrics = get_all_metrics(top_events)
    # ['Embedding/grad norm', 'Embedding/param norm', 'Outer Loss/train', 'Outer Loss/val']
    for m in metrics:
        best_key = None
        if 'Loss' in m:
            best_key = f'Best {m}'
        if 'embedding' in m.lower() and not tailoring:
            continue
        final_metrics.append(track_metric(
            top_events[1:], m, m, writer, best_key=best_key))

    # Add events from each sub folder
    for f in files:
        _f = join(run_dir, f)
        if os.path.isdir(_f):
            folder_files = os.listdir(_f)
            # Expect only one tfevents file per subfolder
            assert len(folder_files) == 1
            events = read_events(summary_iterator(join(_f, folder_files[0])))
            metrics = get_all_metrics(events)
            assert len(metrics) == 1
            step = f.split('_')[-1].split(' ')[0].strip()
            step = int(step)
            metric = f'{metrics[0]}/{step} Steps'
            key = metrics[0]
            best_key = f'Best {metric}'
            if 'inner' in key.lower() and not tailoring:
                continue
            final_metrics.append(track_metric(
                events[1:], key, metric, writer, best_key=best_key))

    final_metrics = reduce(lambda x, y: x | y, final_metrics)
    _hparams = hyperparams(hparams, final_metrics)
    for i in _hparams:
        writer.file_writer.add_summary(i)
    for k, v in final_metrics.items():
        writer.add_scalar(k, v)
    writer.flush()
    writer.close()


def process_dir(log_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    runs = os.listdir(log_dir)
    print('Number of Runs:', len(runs))
    for run in runs:
        if os.path.isdir(join(log_dir, run)):
            process_run(join(log_dir, run), out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    process_dir(args.log_dir, args.out_dir)


if __name__ == '__main__':
    main()
