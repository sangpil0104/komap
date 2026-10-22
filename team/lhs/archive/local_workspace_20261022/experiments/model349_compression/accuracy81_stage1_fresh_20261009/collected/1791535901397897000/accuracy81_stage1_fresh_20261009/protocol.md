# 1단계 R + E 원본 영상 재추론

대표 후보는 E의 Al3Ni/Al 픽셀만 R을 덮는 1단계다. F/확신도 추가 없음. 원본영상 valid20과 test10에서 원래 FP32 릴리스14checkpoint R 및 P100 B4 4fold + HRNet2seed×4fold E를 전부 새로 계산한다. B4 D4/HRflip4/가족FP16반올림/원래팀argmax동률순서를 유지한다. 학습·가중치변경 없음. GPU3의 실제유휴·compute·siblinglock 검사 후 순차 실행, 외부compute 등장시 우리생성그룹만종료, 자동재실행없음. valid의 R/1단계PNG/GT80IoU독립채점, 원래canonical80.4746과차이기록. test정답미사용, 새R/원래exh14마스크와픽셀일치율만기록. 원본부모SHA보존. 반복valid선택/최악영상guard미통과/독립일반화한계 유지. 기존base불변.
