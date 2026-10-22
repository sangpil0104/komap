import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run
from sslab.common import PALETTE, config, digest, read_json
from sslab.data import Patches, gray, labels
from sslab.engine import Context, budget_remaining, load_checkpoint, run_stage, seed_all
from sslab.inference import predict
from sslab.loss import reconstruction_loss
from sslab.metrics import measure, summarize
from sslab.model import SmallUNet, transfer_encoder
from sslab.parallel import DistributedUpdateIndices

torch.set_num_threads(2)


def small_config():
    cfg = config()
    cfg['model']['pretrained'] = False
    cfg['sampling'] = {'core': 32, 'halo': 0}
    cfg['inference'].update(core=32, halo=0)
    cfg['ssl'].update(updates=1, mask_block=4)
    cfg['train'].update(updates=2, effective_batch=1, workers=0, checkpoint_every=2,
                        log_every=1, precision='fp32', cpu_threads=1)
    cfg['control']['max_extra_updates'] = 30
    return cfg


def fixture(root):
    rows = []
    truth = np.tile(np.repeat(np.arange(4), 8), (32, 1)).astype(np.uint8)
    for index in range(100):
        image = np.array([30, 80, 150, 230], dtype=np.uint8)[truth]
        image[index // 32, index % 32] += 1
        source = root / f'original_{index}.png'
        mask = root / f'mask_{index}.png'
        Image.fromarray(image).save(source)
        split = 'train' if index < 70 else 'valid' if index < 90 else 'test'
        if split != 'test':
            Image.fromarray(PALETTE[truth]).save(mask)
        rows.append({'image_id': f'synthetic_{index}', 'source_split': split, 'width': 32, 'height': 32,
                     'image_path': str(source), 'mask_path': str(mask) if split != 'test' else '',
                     'image_sha256': digest(source), 'mask_sha256': digest(mask) if split != 'test' else ''})
    path = root / 'manifest.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


class UnitTests(unittest.TestCase):
    def test_ssl_never_opens_labels_and_only_masks_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'original.png'
            Image.fromarray(np.tile(np.arange(32, dtype=np.uint8), (32, 1))).save(path)
            rows = [{'image_id': 'train', 'source_split': 'train', 'image_path': str(path), 'mask_path': '/does/not/exist'}]
            dataset = Patches(rows, small_config(), 'ssl', 1)
            self.assertNotIn('mask_path', dataset.rows[0])
            with patch('sslab.data.labels', side_effect=AssertionError('SSL read ground truth')):
                image, clean, mask = dataset[0]
            self.assertTrue(mask.sum() > 0)
            self.assertTrue(torch.all(image[0][mask[0] > 0] == .5))
            self.assertTrue(torch.all((mask == 0) | (mask == 1)))
            self.assertEqual(tuple(clean.shape), (1, 32, 32))
            with self.assertRaises(ValueError):
                Patches([{**rows[0], 'source_split': 'valid'}], small_config(), 'ssl', 1)

    def test_loss_gradients_only_on_masked_real_pixels(self):
        logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
        hidden = torch.zeros_like(logits)
        hidden[..., 1, 1] = 1
        loss = reconstruction_loss(logits, torch.ones_like(logits), hidden)
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 1)
        changed = torch.ones_like(logits) * 100
        changed[..., 1, 1] = 0
        self.assertEqual(loss.item(), reconstruction_loss(changed, torch.ones_like(logits), hidden).item())

    def test_transfer_preserves_identical_segmentation_decoder(self):
        seed_all(2026)
        a = SmallUNet()
        b = SmallUNet()
        b.load_state_dict(a.state_dict())
        ssl = SmallUNet(outputs=1)
        with torch.no_grad():
            next(ssl.encoder.parameters()).fill_(.123)
        transfer_encoder(b, ssl.state_dict())
        for name, tensor in a.state_dict().items():
            if not name.startswith('encoder.'):
                self.assertTrue(torch.equal(tensor, b.state_dict()[name]), name)
        self.assertTrue(torch.equal(next(b.encoder.parameters()), next(ssl.encoder.parameters())))
        self.assertFalse(torch.equal(next(a.encoder.parameters()), next(b.encoder.parameters())))

    def test_image_metric_and_absent_false_positive(self):
        cfg = config()['evaluation']
        t1 = np.zeros((10, 10), dtype=np.uint8)
        t2 = np.zeros((1, 1), dtype=np.uint8)
        rows = [measure(t1, t1, cfg), measure(t2, np.ones_like(t2), cfg)]
        summary = summarize(rows, 'one')
        self.assertEqual(summary['image_miou'], 75)
        self.assertEqual(summary['miou_zero'], 12.5)
        self.assertEqual(summary['miou_exclude'], 50)
        self.assertEqual(rows[1]['iou_al3ni'], 0)

    def test_distributed_sampler_exact_batch_including_dummies(self):
        all_indices = []
        for rank in range(6):
            samples = list(DistributedUpdateIndices(2, 8, rank, 6))
            self.assertEqual(len(samples), 4)
            all_indices.extend(index for index, real in samples if real)
        self.assertEqual(sorted(all_indices), list(range(16)))

    def test_native_tiled_inference_covers_border(self):
        class Constant(torch.nn.Module):
            def forward(self, image):
                output = torch.zeros(image.shape[0], 4, *image.shape[-2:])
                output[:, 2] = 10
                return output
        cfg = small_config()
        cfg['inference']['halo'] = 16
        output = predict(Constant(), np.ones((59, 73), np.float32), cfg, torch.device('cpu'))
        self.assertEqual(output.shape, (4, 59, 73))
        np.testing.assert_allclose(output.sum(0), 1, atol=1e-6)
        self.assertTrue(np.all(output.argmax(0) == 2))

    def test_compute_budget_includes_a_prefix(self):
        self.assertEqual(budget_remaining(100, 30, 105), 35)
        self.assertEqual(budget_remaining(100, 10, 20), 0)


class WorkflowTests(unittest.TestCase):
    def test_synthetic_suite_reports_and_cpu_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, rows = fixture(root)
            cfg = small_config()
            cfg['data']['manifest'] = str(manifest)
            ctx = Context(cfg, cpu=True)
            try:
                suite = run.run_suite(cfg, ctx, output_root=root / 'outputs', results_root=root / 'results')
                completed = read_json(suite / 'completed.json')
                self.assertEqual(set(completed['evaluations']), {'A', 'B', 'C'})
                for arm, info in completed['evaluations'].items():
                    self.assertEqual(info['summary']['num_images'], 20)
                    result = Path(info['directory'])
                    self.assertEqual(len(list((result / 'valid_mask_comparison').glob('*.png'))), 20)
                    self.assertTrue(read_json(result / 'verification.json')['saved_panels_verified'])
                    self.assertTrue((result / f'{result.name}.md').exists())
                s = completed['stages']
                self.assertGreaterEqual(s['A_supervised']['compute_seconds'] + s['C_extra']['compute_seconds'],
                                        s['B_ssl']['compute_seconds'] + s['B_finetune']['compute_seconds'])
                self.assertEqual(s['A_supervised']['updates'], s['B_finetune']['updates'])
                self.assertTrue((Path(completed['comparison']) / 'budget.json').exists())
                # Check immutable completion and resume without producing another report.
                before = len(list((root / 'results').iterdir()))
                run.run_suite(cfg, ctx, resume=suite, results_root=root / 'results')
                self.assertEqual(before, len(list((root / 'results').iterdir())))
                # Save after one update, resume, and compare against uninterrupted A.
                (suite / 'SAVE_AND_STOP').touch()
                paused = run_stage(suite, cfg, ctx, 'resume_check', suite / 'initial.pt')
                self.assertIsNone(paused)
                saved = Path(read_json(suite / 'resume_check/latest.json')['checkpoint'])
                self.assertEqual(load_checkpoint(saved)['update'], 1)
                (suite / 'SAVE_AND_STOP').unlink()
                resumed = run_stage(suite, cfg, ctx, 'resume_check', suite / 'initial.pt')
                expected = load_checkpoint(s['A_supervised']['checkpoint'])['model']
                actual = load_checkpoint(resumed['checkpoint'])['model']
                for name in expected:
                    self.assertTrue(torch.equal(expected[name], actual[name]), name)
                self.assertTrue(saved.exists())
            finally:
                ctx.close()
                gray.cache_clear()
                labels.cache_clear()


if __name__ == '__main__':
    unittest.main()
