# R 캐시 재생 대조 분리

첫attempt는canonicalR과A11_step2_sample4 1픽셀차이로중단·보존했다. 이번에는R저장캐시출력exact gate를검사하고,canonicalR과차이모든영상을기록한다. reference25는canonicalR로정확재현하고,r_replay25는동일R캐시재생대조군이다. R경로제거는canonical25/원래27/r_replay25 3개대조모두평균−.01%p/최악−.1%p gate. OpenCV2threads를원래실행과맞춘다. GT없는예측및별도PNG채점/test미사용/반복valid한계.
