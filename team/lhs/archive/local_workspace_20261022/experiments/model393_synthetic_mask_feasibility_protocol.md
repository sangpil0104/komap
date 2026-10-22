# Model393: synthetic phase-mask feasibility screen (CPU, pre-analysis)

## Question and scope

This is a descriptive go/no-go screen for a possible, later synthetic
microstructure image-mask experiment. It is not model training, segmentation
evaluation, or evidence of an mIoU gain. It reads **only** the masks contained
in the 70-image `train.zip`; it does not read image pixels, valid/test data,
Model382/392 checkpoints, OOF probabilities, or any model score. The experiment
does not authorize GPU work or change the locked Model382/392 recipes.

The only output is a fixed JSON inventory of phase occurrence by specimen
group, per-image and per-group phase fractions, connected-component scales,
Eutectic-Si distance-transform width proxies, immediate phase-contact
frequencies, and short-range two-point mask correlations. Empty or invalid
labels, duplicate IDs, or a train-image count other than 70 fail closed.

## Frozen interpretation

- A phase represented in fewer than five independent specimen groups is
  unsupported for a morphology-conditioned generator in this dataset.
- Eutectic-Si width proxies and component statistics describe annotation scale;
  they are not physical ground truth, noise estimates, or an upper bound.
- Between-group variation is reported separately from within-group image
  variation. No group is dropped or reweighted after inspecting its values.
- The only hard gate in this screen is the five-group phase-support rule above.
  Width, connectivity, and contact summaries do not get converted into an
  automatic GO/NO-GO after seeing their values. Passing the support gate means
  only “worth designing a separate feasibility pilot”; it is not permission to
  train. A separate fold-local, image-to-mask generator protocol and source
  lock are required before any training. A proposed candidate would still need
  a paired grouped-OOF experiment and must not use held-out fold masks to fit
  its mask generator, renderer, normalization, or style bank.
- Because this exploratory audit summarizes all train masks, any experiment
  whose design is materially adapted to its findings is selection-conditional;
  it cannot be described as an untouched confirmatory test.

## Motivation and limits

Materials literature has explored simulated morphology plus image-style
translation for microscopy augmentation, and synthetic pretraining for uranium
oxide microstructure segmentation. A recent electron-micrograph pipeline also
generated mask-image pairs from morphology priors. These are adjacent evidence,
not validation for four-phase Al-Si optical micrographs. In particular, coupled
Eutectic-Si morphology and Al3Ni adjacency may not be represented by a simple
Potts/particle model.

- Ma et al., [Data augmentation in microscopic images for material data mining](https://arxiv.org/abs/1905.04711).
- Ly et al., [Improving Microstructures Segmentation via Pretraining with Synthetic Data](https://www.pnnl.gov/publications/improving-microstructures-segmentation-pretraining-synthetic-data).
- Nguyen et al., [Generative learning of morphological and contrast heterogeneities for self-supervised electron micrograph segmentation](https://www.nature.com/articles/s41524-025-01800-5).
