# 파일 검증 및 출처

그림18개와표11개, README의SHA-256을 기록했다. 이 파일 자체는 목록에서 제외한다.

| 파일 | 크기(bytes) | SHA-256 |
| --- | --- | --- |
| README.md | 4839 | 3de45a0b5e4b988c5d862d60e44c35574dccb1ba0aa5c5dcce69a89226a4de9a |
| figures/01_dataset_overview.png | 150936 | 785df201ac71714451186e8936a37ce5fa8c53671c1f071cddfe9b5f25eba33c |
| figures/02_image_resolution.png | 129752 | 3f1768b285725f8dcd91deca5ac3c2b18d00a329a5542fdb0ebae80abafda179 |
| figures/03_class_pixel_share.png | 156868 | 61d0e2d6659d82d079cf20c5ec95c100ed17a2fb49a491cba2f5187264b39db4 |
| figures/04_class_image_area.png | 271777 | ef80e0477f280c1b0bccd8fa57091ba9adcb47423c8333d9eaa997f52801e3eb |
| figures/05_brightness_distribution.png | 506963 | a1a20c86ca83e344c74ec0354e9be730fb9884aec50185ab86c3e87c6ebc42de |
| figures/06_specimen_overlap.png | 104336 | ed20f4f2d837237e6829e05776fbdcc00a69cb48433a9c9240ca7da775f5ffe4 |
| figures/07_raw_zscore_cdf.png | 5472417 | 56a7004b86d8828d19827e5bbeb2026419a4002155df39d7b48d4f4df3783067 |
| figures/08_preprocessing_flow.png | 351005 | aecf401cf07d221fcd8af637349f6f1b33059e8f8f96658f248c6f8fb1b82a79 |
| figures/09_model349_pipeline.png | 354820 | b58fb9721c7f4f7af621c47ece041e88eceed28ef16cc585d9d3c01536890c36 |
| figures/10_resnet_aspp_native.png | 366855 | 2f0c2b2d9005f6e3e455d5c2f7a1ca4947fa0ff3fc4b3e6ca9cb628dadb10682 |
| figures/11_b4_hrnet_architecture.png | 319083 | c5e387a1ae8961cdddbb0267d28db213aa5de06f8cdc7aa6efbc75816d418d30 |
| figures/12_classwise_fusion.png | 320174 | d5962a5539b31ec5457db006e3afd710c47de03160e2674bf410e94f486720cc |
| figures/13_stage_performance.png | 194334 | 067da9124e7422585f7a74aec5f2b35f5f06e7930d6ab94fe8d2ec3c11f61ed1 |
| figures/14_class_iou.png | 153391 | 6e66d2267f6289ce35224fab56510eb34a5da0a7388c16c190c37728eab98a8b |
| figures/15_per_image_miou.png | 364593 | d1bfdc3b79a4c384c501295d3ee38ec4d206fe01ef6be5b308a5d1aab42511dd |
| figures/16_per_image_class_iou.png | 518876 | b83cec5e2cfdb81f1ab776acc8c944265618387bb3a14c52765bf651f4a76b74 |
| figures/17_frozen_cache_ablation.png | 279587 | 3978b6e0ae82f0e428744aab0ec8db0a0d0aa3c52f541ca55d25f544088e6e4d |
| figures/18_image_miou_definition.png | 192598 | 175b874973daead2128b8402a143097b73d43cc628069fd8ef1626a9b804e9eb |
| tables/01_dataset_summary.md | 1667 | 3432912cf89ad7e114f5ad230b36749369e0933a8831d6cc844e884f0432e2ed |
| tables/02_class_distribution.md | 1766 | b5e3243bce97a380f0ca0bd0184e40924ef934a3196c4249ccc4ff86e11f8912 |
| tables/03_preprocessing_and_augmentation.md | 2435 | 44e76b4da329fc1635dd6dc1c8ac2c55af4f216c6eb1617f55e5cf6dfbdc09b4 |
| tables/04_model_components.md | 2184 | 7ee14cf002aed80bfc470242d3a6f8a9fb505e7266f1f9545a97d89a9e33e923 |
| tables/05_training_and_inference.md | 2603 | c7eec9fef44ab9368d1d8bbb9fac4c43c8f51ab101edd75701d1f24d66bbeb92 |
| tables/06_fusion_and_postprocessing.md | 2123 | e8b39dbc23360852a12ba9b64c51879f45f2caa281e7aa6c78fa03b76842c66d |
| tables/07_stage_performance.md | 1140 | fea0723b0cc09a166724f55c0c65b801366a9ec6a67cdd3fc30a76447e61f279 |
| tables/08_validation_scores.md | 2553 | a6dd84940ca259558f012f48871a12e7c639ce7fe3132eb57ba54489e174049b |
| tables/09_frozen_probability_comparisons.md | 3736 | 83e578c5af98f450786be2d255244ee239c1967536cef74233ebc64ff0a2121e |
| tables/10_evaluation_and_reproducibility.md | 2773 | 39f3cfe18c178f1d9cc1ebbd4caef110b7e8df0d2bf0e35237217a2d746032e7 |
| tables/11_figure_captions_and_page_plan.md | 5859 | 12bd8ead8d5a4e79331c03720425f6537a2321e8c2d0c2a069dd6e24b66f39c0 |

분석 기준 실행: `/Data3/sppark/komap/experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z`.

모든 원래 확률결합·후처리 조건에서 기존 valid 마스크20장과 픽셀 일치를 검증했다. 입력 마스크에서알려지지 않은 색상이 없는지 확인했다. Global mIoU는 계산하지 않았다.

확률 캐시 입력 파일의 해시 기록은 프로젝트 내 `experiments/model349_report_materials_v1/outputs/analysis_20261007_1/sources.json`에 보존했다.
