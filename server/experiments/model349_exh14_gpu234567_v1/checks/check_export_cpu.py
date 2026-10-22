"""Synthetic evaluation output integration check; never calls a model."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, SOURCE, inference, write_json
import export_results as exporter


def main():
    config = inference().load_config(SOURCE / 'configs/model349_lite.yaml')
    palette = np.asarray(config['palette'], dtype=np.uint8)
    target_small = np.repeat(np.array([0, 2, 3], np.uint8), 9).reshape(3, 9)
    prediction_small = target_small.copy()
    prediction_small[0, 0], prediction_small[1, 0], prediction_small[2, 0] = 2, 3, 0
    np.testing.assert_allclose(exporter.class_ious(prediction_small, target_small), [.8, 0, .8, .8])
    assert np.mean(exporter.class_ious(prediction_small, target_small)) == .6000000000000001
    assert np.mean(exporter.class_ious(target_small, target_small)) == .75
    try:
        exporter.class_ious(np.array([[255]], np.uint8), np.array([[0]], np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid labels were accepted.')

    with tempfile.TemporaryDirectory(prefix='exh14_export_') as directory:
        temporary = Path(directory)
        valid = temporary / 'valid'
        (valid / 'images').mkdir(parents=True)
        (valid / 'masks').mkdir()
        run = temporary / 'run'
        work = run / 'inference'
        (work / 'predicted_labels').mkdir(parents=True)
        (work / 'predicted_masks').mkdir()
        snapshot = run / 'source_snapshot'
        snapshot.mkdir()
        for name in ['provenance.json', 'local_weights.json', 'data_manifest.json']:
            write_json(snapshot / name, {'synthetic_test_only': True})
        rows, scores = [], []
        for index in range(20):
            # Different image sizes ensure the exported metric is an image average.
            repeats = (10, 10) if index < 10 else (20, 20)
            target = np.tile(target_small, repeats)
            prediction = target.copy() if index < 10 else np.tile(prediction_small, repeats)
            name = f'synthetic_{index:02d}'
            original = Image.new('RGB', (target.shape[1], target.shape[0]), (91, 91, 91))
            path = valid / 'images' / f'{name}_image.png'
            original.save(path)
            Image.fromarray(palette[target]).save(valid / 'masks' / f'{name}_mask.png')
            Image.fromarray(prediction).save(work / 'predicted_labels' / f'{name}.png')
            Image.fromarray(palette[prediction]).save(work / 'predicted_masks' / f'{name}_mask.png')
            rows.append(dict(name=name, path=str(path), size=list(original.size)))
            ious = [.9999999999999999, 0, 1, 1] if index < 10 else [.8, 0, .8, .8]
            scores.append(dict(image=name, class_iou=ious, image_miou=.75 if index < 10 else .6))
        write_json(work / 'completion.json', dict(images=20, elapsed_seconds=0.0))
        write_json(work / 'input_manifest.json', dict(config=config, images=rows))
        metric = dict(count=20, images=scores, image_miou=.675)
        write_json(work / 'metrics.json', metric)
        old_valid = exporter.VALID
        exporter.VALID = valid
        try:
            first = exporter.export(run, temporary / 'results')
            assert abs(json.loads((first / 'metrics.json').read_text())['image_miou'] - .675) < 1e-12
            assert len(list((first / 'valid_mask_comparison').glob('*.png'))) == 20
            assert (first / f'{first.name}.md').is_file()
            assert (first / 'complete.json').is_file()
            second = exporter.allocate_result(temporary / 'results')
            assert first != second and first.is_dir()
            metric['image_miou'] = .9
            write_json(work / 'metrics.json', metric)
            try:
                exporter.export(run, temporary / 'results')
            except ValueError:
                pass
            else:
                raise AssertionError('Incorrect upstream metric was accepted.')
        finally:
            exporter.VALID = old_valid

    # A readable full-size sample using a local original image and synthetic prediction.
    path = sorted((exporter.VALID / 'images').glob('*.png'))[0]
    name = path.stem.removesuffix('_image')
    target = inference().decode_gt(exporter.VALID / 'masks' / f'{name}_mask.png', config['palette'])
    prediction = target.copy()
    prediction[100:160, 100:240] = (target[100:160, 100:240] + 1) % 4
    score = float(np.mean(exporter.class_ious(prediction, target)))
    with Image.open(path) as original:
        panel = exporter.comparison(original, prediction, target, palette, 'SYNTHETIC LAYOUT CHECK - NOT A MODEL RESULT', score)
        width, height = original.size
    pixels = np.asarray(panel)
    expected_error = np.full((*target.shape, 3), 255, np.uint8)
    bad = prediction != target
    expected_error[bad] = palette[target[bad]]
    np.testing.assert_array_equal(pixels[114:114 + height, width:2 * width], expected_error)
    bottom = 80 + height + 2 * 34
    np.testing.assert_array_equal(pixels[bottom:bottom + height, :width], palette[prediction])
    np.testing.assert_array_equal(pixels[bottom:bottom + height, width:2 * width], palette[target])
    panel.save(ROOT / 'checks/layout_preview_synthetic.png')
    write_json(ROOT / 'checks/export_cpu.json', dict(ok=True, synthetic_only=True, training_started=False,
        inference_started=False, metric_example_percent=60.0, mean20_percent=67.5,
        panel_pixel_layout_verified=True, empty_class_zero=True, invalid_labels_rejected=True,
        upstream_metric_mismatch_rejected=True, unique_results=True, exported_panels=20))
    print('PASS: image mIoU, empty classes, 20-panel export, error colors/layout, new result numbering.')


if __name__ == '__main__':
    main()
