"""Suite protocol/failure isolation tests; detector and GPU are never invoked."""
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'colab_support'))
from comparison_suite import representative_offsets, run_short_suite


class ComparisonSuiteTests(unittest.TestCase):
    def test_representative_offsets_use_real_source_frames(self):
        info = {'fps': 30000 / 1001, 'frames': 3855}
        offsets = representative_offsets(info)
        self.assertEqual([round(value * info['fps']) for value in offsets], [0, 1897, 3795])
        self.assertTrue(all(round(start * info['fps']) + round(2 * info['fps']) <= 3855
                            for start in offsets))
        self.assertEqual(representative_offsets({'fps': 5, 'frames': 10}), [0])
        with self.assertRaises(ValueError):
            representative_offsets({'fps': 30, 'frames': 20})

    def run_fixture(self, failed=False):
        import chunked_pipeline as cp
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.pt'
            model.write_bytes(b'one shared model')
            paths = [root / f'video_{index}.mp4' for index in range(5)]
            for index, path in enumerate(paths):
                path.write_bytes(f'source {index}'.encode())
            result = {'status': 'technical_checks_passed_review_required',
                      'baseline': {'unique_tracker_ids_not_player_count': 7},
                      'improved': {'unique_tracker_ids_not_player_count': 6}}
            calls = []
            def compare(path, output, config, **kwargs):
                calls.append((path, output, config, kwargs))
                if failed and len(calls) == 2:
                    raise RuntimeError('GPU failure fixture')
                return result
            with patch.object(cp, 'MODEL_PATH', str(model)), \
                    patch('colab_helpers.video_info', return_value={'fps': 60, 'frames': 600}), \
                    patch('short_validation.run_short_comparison', side_effect=compare):
                summary = run_short_suite(paths, root / 'suite',
                    lambda video: {'filter_field': False, 'enable_ocr': False})
                persisted = json.loads((root / 'suite' / 'suite_summary.json').read_text())
                self.assertEqual(persisted, summary)
                self.assertEqual(len(calls), 15)
                self.assertEqual(len({str(call[0]) for call in calls}), 5)
                self.assertTrue(all(call[3]['require_gpu'] for call in calls))
                self.assertTrue(all(call[3]['duration_s'] == 2 for call in calls))
                self.assertTrue(all(call[3]['chunk_size'] == 30 for call in calls))
                self.assertFalse(summary['manual_review_approved'])
                self.assertFalse(summary['accuracy_validated'])
                self.assertIsNone(summary['protocol']['baseline_configuration'])
                return summary

    def test_same_baseline_protocol_for_all_five_videos(self):
        summary = self.run_fixture()
        self.assertEqual(summary['status'], 'technical_checks_passed_review_required')

    def test_failure_is_retained_and_other_clips_continue(self):
        summary = self.run_fixture(failed=True)
        self.assertEqual(summary['status'], 'validation_failed')
        self.assertEqual(sum(row['status'] == 'failed' for row in summary['clips']), 1)
        self.assertIn('GPU failure fixture', summary['clips'][1]['error'])

    def test_wrong_video_count_fails_before_output_or_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'suite'
            with self.assertRaisesRegex(ValueError, 'exactly 5'):
                run_short_suite([], output, lambda video: {})
            self.assertFalse(output.exists())
