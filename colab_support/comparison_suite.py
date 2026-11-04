"""The same short baseline/improved protocol for every selected source video.

Only technical success is aggregated. Each clip still needs identity review;
the suite never starts full processing or grants per-video approval.
"""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def representative_offsets(info, duration_s=2):
    """Opening/middle/late clips, on exact source frames; deduplicate short inputs."""
    fps, frames = float(info['fps']), int(info['frames'])
    if fps <= 0 or not 0 < duration_s <= 5:
        raise ValueError('Need positive FPS and a duration of at most five seconds')
    count = round(fps * duration_s)
    if count < 10 or count > frames:
        raise ValueError('Video must contain a complete short clip of at least ten frames')
    last = frames - count
    return [index / fps for index in sorted({0, last // 2, last})]


def run_short_suite(video_paths, output_dir, configuration_for_video,
                    duration_s=2, require_gpu=True, min_available_gb=2.0,
                    expected_videos=5):
    """Run three short windows per video; preserve failures and continue the suite.

    configuration_for_video(path) returns the full-source improved config. The
    short runner remaps calibrated source keyframes to each sampled window.
    Every pair uses seed=0, the same model bytes and the same baseline settings.
    """
    import colab_helpers as ch
    import chunked_pipeline as cp
    from colab_session import short_chunk_size
    from short_validation import run_short_comparison

    paths = [Path(path).resolve() for path in video_paths]
    if len(paths) != expected_videos or len(set(paths)) != len(paths):
        raise ValueError(f'Select exactly {expected_videos} distinct source videos')
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError('Every selected source video must exist')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    model = Path(cp.MODEL_PATH).resolve()
    model_hash = file_sha256(model)
    protocol = {'model_sha256': model_hash, 'model_name': model.name,
                'baseline_configuration': None, 'seed': 0,
                'baseline_confidence_blurred': .22, 'baseline_confidence_sharp': .25,
                'iou': .35, 'imgsz': 1280, 'duration_s': duration_s,
                'windows': ['opening', 'middle', 'late'], 'require_gpu': require_gpu}
    summary = {'protocol': protocol, 'expected_videos': expected_videos,
               'status': 'running', 'clips': [], 'manual_review_approved': False,
               'accuracy_validated': False}
    destination = root / 'suite_summary.json'

    def save():
        destination.write_text(json.dumps(summary, indent=2, default=str) + '\n', encoding='utf-8')

    save()
    for video_index, path in enumerate(paths):
        source_hash = file_sha256(path)
        try:
            info = ch.video_info(path)
            offsets = representative_offsets(info, duration_s)
            config = configuration_for_video(path)
        except Exception as exc:
            summary['clips'].append({'source': str(path), 'status': 'failed',
                                     'error': f'{type(exc).__name__}: {exc}'})
            save()
            continue
        for window_index, start_s in enumerate(offsets):
            target = root / f'video_{video_index + 1:02d}_clip_{window_index + 1:02d}'
            row = {'source': str(path), 'source_sha256': source_hash,
                   'start_s': start_s, 'duration_s': duration_s, 'directory': str(target)}
            print(f'Short suite {video_index + 1}/{len(paths)}: {path.name}, start {start_s:.3f}s')
            try:
                if file_sha256(model) != model_hash:
                    raise RuntimeError('Model changed during comparison suite')
                result = run_short_comparison(path, target, config,
                    chunk_size=short_chunk_size(info, duration_s), duration_s=duration_s,
                    start_s=start_s, require_gpu=require_gpu, min_available_gb=min_available_gb)
                if file_sha256(path) != source_hash or file_sha256(model) != model_hash:
                    raise RuntimeError('Source video or model changed during comparison')
                if result.get('status') != 'technical_checks_passed_review_required':
                    raise RuntimeError('Short comparison did not pass its technical checks')
                row['status'] = result['status']
                row['metrics'] = {label: {name: result[label].get(name) for name in (
                    'unique_tracker_ids_not_player_count', 'unknown_team_observations',
                    'unknown_jersey_number_observations', 'true_id_switches')}
                    for label in ('baseline', 'improved')}
            except Exception as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            summary['clips'].append(row)
            save()
    summary['status'] = ('technical_checks_passed_review_required'
        if summary['clips'] and all(row['status'] == 'technical_checks_passed_review_required'
                                   for row in summary['clips']) else 'validation_failed')
    save()
    return summary
