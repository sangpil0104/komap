"""Validate the real upstream naming contract without renaming its outputs."""
from pathlib import Path

def validate_prediction_inventory(images, output, expected_count=None):
    images,output=Path(images),Path(output)
    inputs=list(images.glob('*_image.png'))
    names={p.stem.removesuffix('_image') for p in inputs}
    if not names or len(names)!=len(inputs):raise ValueError('Empty or duplicated input image inventory')
    if expected_count is not None and len(names)!=expected_count:
        raise ValueError(f'Expected {expected_count} images, got {len(names)}')
    for directory,suffix in [('predicted_labels','.png'),('predicted_masks','_mask.png')]:
        expected={name+suffix for name in names}
        actual={p.name for p in (output/'fusion'/directory).glob('*.png')}
        if actual!=expected:
            raise ValueError(f'Incomplete prediction inventory: {directory}; missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}')
    if not (output/'completion.json').is_file():raise ValueError('Missing prediction completion receipt')
    return dict(ok=True,images=len(names),labels_suffix='.png',masks_suffix='_mask.png')
