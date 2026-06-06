# labs-clip-vision
onnx와 rknn으로 성능을 확인하는 실험


### utils

model_download.py: 허깅페이스에서 clip-vit 모델을 다운로드하는 스크립트

``` bash
uv run src/utils/model_download.py
```
다운로드 모델은 models/onnx 폴더에 저장

### data

imgs/ 샘플 이미지 보관소

violence: 폭행 행동의 이미지 리스트로 20장씩 정리
normal: 폭행처럼 프레임의 변화량이 크지만 일반 행동 (장난, 물건 쏟음 등) 이미지 리스트를 20장씩 정리
bg: 사람도 없고 변화도 없는 완벽한 배경 이미지 리스트를 20장씩 정리
