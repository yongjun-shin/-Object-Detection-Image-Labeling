# ObjectDetection-Image_Labeling

## 이미지 주석화 작업 가이드라인
- 작성일 : 2021.07.22
- 작성자 : 신용준

***

### 목차
1. 프로그램 설치
* 1-1. anaconda(& jupyter)
* 1-2. labelme
* 1-3. labelimg
2. 이미지 주석화(데이터셋 만들기)
* 2-1. Polygon(labelme)
* 2-2. Bounding box(labelImg)
3. 훈련(학습)
* 3-1. Polygon
* 3-2. Bounding box
4. 정확도 예측
* 4-1. Polygon(삭제)
* 4-2. Bounding box
* 4-3. 비교
5. 이미지 증강

***

### 1. 프로그램 설치
> 1-1. anaconda(& jupyter)
- https://www.anaconda.com/ 클릭

- 왼쪽 상단 Products 하위 메뉴에 Individual Edition 클릭.

![image](https://user-images.githubusercontent.com/73512218/192139675-6045af2b-cee5-46b9-b488-5fe6f8183815.png)

- Download 버튼을 클릭하여 설치 파일을 다운로드. 이 때, 자신의 운영체제 확인할 것.

![image](https://user-images.githubusercontent.com/73512218/192139737-e5b26bfd-c7b1-4dcd-aa28-a93d7fed7fa0.png)

- 다운로드 된 파일 실행. 이후 전부 Next와 같은 동의의 의미를 가지는 버튼 클릭.

![image](https://user-images.githubusercontent.com/73512218/192139750-c636dc70-ab73-4917-8f38-bc83800505fb.png)

- 이 부분은 크게 중요하지 않다. 필요한 것을 선택하여 Next 클릭.

![image](https://user-images.githubusercontent.com/73512218/192139767-fcc25fa0-eee9-4544-ae32-647024d81f99.png)

- “Add Anaconda3 to the system PATH environment variable” 클릭

![image](https://user-images.githubusercontent.com/73512218/192139786-a367ed52-0c71-47dd-b41f-521fc9faa0b5.png)

- 설치 완료 후 ‘시작’ 버튼 -> Anaconda Navigator 실행 -> Jupyter ‘launch’ 클릭

![image](https://user-images.githubusercontent.com/73512218/192139799-3cc7bf65-36b3-4c96-9b6a-0d77c2f30136.png)

> 1-2. labelme
- Anaconda Prompt 관리자 권한으로 실행하기

![image](https://user-images.githubusercontent.com/73512218/192139835-7760aa2d-37d0-4ef1-9a51-4cea6af4b598.png)

- Anaconda Prompt에 conda create --name=labelme python=3.6 입력하면 Proceed([y]/n)? 이 뜨는데 y를 입력 후 엔터를 눌러준다.

![image](https://user-images.githubusercontent.com/73512218/192139843-97734544-f4ec-4ff7-9e50-bc775924c9eb.png)

- 설치가 완료되면 conda activate labelme를 입력 후 pip install labelme 입력한다.

![image](https://user-images.githubusercontent.com/73512218/192139853-a569f802-cc41-48ae-9fd6-82693cac4659.png)
![image](https://user-images.githubusercontent.com/73512218/192139857-5b7ebec9-5b2f-4ef4-9418-1b957b3a94cb.png)

- 설치 완료 후 labelme 가상환경인 상태에서 labelme 입력해주면 labelme 창이 나타난다.

> 1-3. labelimg
- Anaconda Prompt창에서 git clone https://github.com/tzutalin/labelImg.git 입력

![image](https://user-images.githubusercontent.com/73512218/192139892-4de3a244-9ea2-40b9-8185-70d89a8dc854.png)

- 받은 github 폴더로 접근을 위해 cd labelImg 입력

- 필요한 라이브러리 다운로드 위해 각 명령을 한 줄씩 차례대로 입력해준다.

![image](https://user-images.githubusercontent.com/73512218/192139910-20f0ab25-1666-4d1a-b460-1dcf7c9e12a6.png)

- 다운로드가 모두 완료되면 python labelImg.py 입력하여 labelImg 창을 실행시킨다.

### 2. 이미지 주석화(데이터셋 만들기)
> 2-1. Polygon(labelme)
- labelme 가상환경에서 실행할 수 있다. Anaconda prompt에서 conda activate labelme 입력하거나 Anaconda Navigator에서 Environments -> labelme -> ▶ -> Open terminal 클릭하여 가상환경 구축 가능하다.

![image](https://user-images.githubusercontent.com/73512218/192139935-621451c6-2519-42ea-ab0c-8f9d6af10d6d.png)
![image](https://user-images.githubusercontent.com/73512218/192139942-febb724f-e459-4348-8426-5f74bfb4a828.png)

- prompt에서 labelme라고만 입력해주면 labelme 편집 창이 나온다.

![image](https://user-images.githubusercontent.com/73512218/192139954-dab86fc8-a079-460b-a5f4-fd9973529d56.png)

- Open 클릭 후 이미지 주석화하기 원하는 이미지 파일을 선택하거나 Open Dir을 통해 이미지 파일들의 경로를 선택할 수 있다.

- 이미지가 load되면 Create Polygons를 클릭한 후 polygon 방식으로 주석화 할 수 있다.

![image](https://user-images.githubusercontent.com/73512218/192139968-31f8acef-a229-4348-8501-2cb015cee897.png)
![image](https://user-images.githubusercontent.com/73512218/192139973-b834f2d5-8053-46af-8655-8f16e08ea1d4.png)

- Save 버튼을 눌러 .json 형식의 파일로 저장할 수 있다.

- Anaconda prompt창 열어서 pyqt5와 labelme, git 패키지 설치
```python
pip install pyqt5
pip install labelme
conda install git
```

- 현재 경로에 임의의 폴더 data 생성후 cd data로 경로이동 후 git clone
  - git clone https;//github.com/Tony607/labelme2coco.git

- data 폴더안에 생성된 labelme2coco 폴더 안의 파일을 다지운다. 그 후 라벨링한 사진 파일과 결과로 나온 json 파일을 다 이폴더에 넣는다.

![image](https://user-images.githubusercontent.com/73512218/192140088-d226fa09-30ea-4508-b72d-41c6fe95f0c5.png)

- cd labelme2coco 입력하여 경로 이동 후 다음 코드를 입력하여 훈련을 시작한다.
```python
python labelme2coco.py images
```
  - 그 결과로 trainval.json 파일이 생성되었다.

> 2-2. Bounding box(labelImg)
- 가상 환경 따로 설정해주지 않고 기본 Anaconda Prompt 창에서 labelImg 입력하면 바로 labelImg 창이 열린다. 

![image](https://user-images.githubusercontent.com/73512218/192140116-a5021ac5-7b36-407a-b1f6-a5600cbd9e18.png)

- 이전 labelme와 동일하게 이미지 파일이나 파일들의 경로를 load 해준다. (Open 클릭)

- 표시된 부분이 PascalVOC라고 되어있으면 한 번 더 클릭하여 yolo로 바꿔준다.(형식)

![image](https://user-images.githubusercontent.com/73512218/192140133-ee7c9add-7616-4369-a160-9fb00f40c154.png)

- Create RectBox 클릭 후 객체를 다 담는 가장 작은 크기의 사각형을 만들어준다.

![image](https://user-images.githubusercontent.com/73512218/192140140-107e90f6-a664-45a5-814f-d774908093df.png)

- Change Save Dir을 누르면 라벨링이 완료된 파일들이 저장될 경로를 미리 지정할 수 있다. 이 후 save 버튼을 눌러서 .txt 형식의 파일로 저장해준다

### 3. 훈련(학습)
> 3-1. Polygon
- Chrome에서 구글 드라이브 -> 내 드라이브 -> 더보기 -> Google colaboratory 클릭

![image](https://user-images.githubusercontent.com/73512218/192140460-b7a6bf93-4252-488e-aa5a-fefeb8f9662b.png)

- 화면 상단의 런타임 -> 런타임 유형 변경 -> 하드웨어 가속기를 None에서 GPU로 변경.

- 왼쪽 폴더 아이콘 -> 드라이브 마운트 -> Google Drive에 연결 클릭

![image](https://user-images.githubusercontent.com/73512218/192140481-0ffbe985-feaa-42a3-8bb7-e948a4897836.png)

- content -> drive -> MyDrive에 임의의 폴더 생성 (ex. data), 폴더 우클릭 -> 경로복사 -> 코드 첫 줄에 %cd + (Ctrl + V) 입력 후 실행

![image](https://user-images.githubusercontent.com/73512218/192140493-bf6112cf-bd4e-452a-84cf-12173f7f5e89.png)

- torchvision 설치 (아래 코드 입력 후 실행)
```python
!pip install -U torch torchvision
!pip install git+https;//github.com/facebookresearch/fvcore.git
import torch, torchvision
torch.__version__
```

- detectron 오리지널 github clone (%cd 경로는 본인 경로에 맞게 수정해야합니다.)
```python
%cd /content/drive/MyDrive/data
!git clone https://github.com/facebookresearch/detectron2 detectron2_repo
!pip install -e detectron2_repo
```

- 상단의 런타임 -> 런타임 다시 시작 클릭

- 라이브러리 import
```python
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
  
import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
  
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
```

- 내가 만든 데이터셋 폴더 생성 -> data2/images, images안에는 사진(.JPG) 넣고 data2 폴더에는 trainval.json 파일 넣는다
```python
%cd /content/drive/MyDrive/data
!mkdir data2
%cd data2
!mkdir images
%cd ..
!pwd
```

![image](https://user-images.githubusercontent.com/73512218/192140509-7b6a8106-50b0-48f3-8147-c707d410a3d1.png)

![image](https://user-images.githubusercontent.com/73512218/192140520-a0abf28b-bd4f-4017-b8ca-74d7da98a55a.png)

- gsm(건숙문) 객체를 추가
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances(“gsm”, {}, “./data2/trainval.json”, “./data2/images”)
person_metadata = MetadataCatalog.get(“gsm”)
dataset_dicts = DatasetCatalog.get(“gsm”)
```

- 레이블 데이터 체크
```python
import random
for d in random.sample(dataset_dicts, 3):
	img = cv2.imread(d[“file_name”])
	visualizer = Visualizer(img[:, :, ::-1], metadata=person_metadata, scale=0.5)
	vis = visualizer.draw_dataset_dict(d)
	cv2_imshow(vis.get_image()[:, :, ::-1])
```

![image](https://user-images.githubusercontent.com/73512218/192140797-7fd842e8-b1bb-4ec0-b33c-16824c330e73.png)

위 사진은 참고용 사진이고 이와 같이 라벨링이 잘 되었는지 눈으로 확인할 수 있다.

- Training 시작 - 완료되면 data/output 폴더에 모델 가중치가 저장된다.
```python
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file(“./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml”)
cfg.DATASETS.TRAIN = (“gsm”)
cfg.DATASETS.TSET = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS=“detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl”
cfg.SOLVER.IMS_PER_BATCH = 2

cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 300
train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

> 3-2. Bounding box
- Chrome에서 구글 드라이브 -> 내 드라이브 -> 더보기 -> Google colaboratory 클릭

![image](https://user-images.githubusercontent.com/73512218/192140881-0f4b045d-67de-4c71-94c5-2071ae9fd5ea.png)

- 화면 상단의 런타임 -> 런타임 유형 변경 -> 하드웨어 가속기를 None에서 GPU로 변경.

- 왼쪽 폴더 아이콘 -> 드라이브 마운트 -> Google Drive에 연결 클릭

![image](https://user-images.githubusercontent.com/73512218/192140905-4fe84559-1baf-42b1-b01f-95b943da1eee.png)
또는

![image](https://user-images.githubusercontent.com/73512218/192140918-cde97b42-36de-4319-9648-a83b3530f5c2.png)

이러한 코드를 실행하여 뜨는 브라우저에 들어가서 나오는 코드를 복사하여 입력하면 된다.

- content -> drive -> MyDrive 안에 임의의 폴더 ‘chma’ 생성 후 그 안에 ‘export’ 라는 이름의 폴더를 생성해준다. 또한 chma 안에 data.yaml 이라는 파일을 만들어주고 파일의 내용은 다음과 같다.

![image](https://user-images.githubusercontent.com/73512218/192140927-5dceb5d9-4eaf-4933-8f0a-5b1deb6e1ae2.png)
![image](https://user-images.githubusercontent.com/73512218/192140929-38506657-ce1d-4712-91fa-3d9fc6c0413f.png)

- ‘export’ 폴더 안에 images 폴더와 labels 폴더를 만들어준 후, images 안에는 사진 파일들을 넣어주고 labels 안에는 그 사진 파일들을 라벨링한 텍스트 파일들을 넣어준다.

- YOLOv5 모델을 Github에서 Clone해서 받아온다. 
```python
%cd /content
!git clone https://github.com/ultralytics/yolov5.git
```

- YOLOv5 모델을 사용하기 위한 패키지 설치
```python
%cd /content/yolov5/
!pip install -r requirements.txt
```

- 라벨링에 사용한 사진들을 ‘img_list_co’에 넣어준다.
```python
%cd /
from glob import glob
img_list_co = glob(‘/content/drive/MyDrive/chma/export/images/*’)
print(len(img_list_co))
```

- ‘img_list_co’에 들어있는 이미지를 test(20%), train(80%) dataset으로 나눠준다
```python
from sklearn.model_section import train_test_split
train_img_list_co, val_img_list_co = train_test_split(img_list_co, test-size=0.2, random_state=2000)
print(len(train_img_list_co), len(val_img_list_co))

with open(‘/content/drive/MyDrive/chma/export/train.txt’, ‘w’) as f:
	f.write(‘\n’, join(train_img_list_co)+‘\n’)
with open(‘/content/drive/MyDrive/chma/export/val.txt’, ‘w’) as f:
	f.write(‘\n’, join(val_img_list_co)+‘\n’)
```

- ‘data.yaml’ 파일을 이용하여, train에는 train.txt를 val에는 val.txt를 넣어준다.
```python
import yaml
with open('/content/drive/MyDrive/guide_cocacola/guide_cocacola.yaml', 'r') as f:
	data = yaml.load(f)
print(data)
data['train'] = '/content/drive/MyDrive/guide_cocacola/train.txt'
data['val'] = '/content/drive/MyDrive/guide_cocacola/val.txt'
with open('/content/drive/MyDrive/guide_cocacola/guide_cocacola.yaml', 'w') as f:
	yaml.dump(data, f)
print(data)
```

- 훈련
```python
%cd /content/yolov5/
!python train.py --batch 16 --epochs 50 --data /content/drive/MyDrive/chma/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name results
```

![image](https://user-images.githubusercontent.com/73512218/192141013-7e5c9ca4-9679-4936-abde-db2cbcc8df7b.png)

### 4. 정확도 예측
> 4-1. Polygon(삭제)

> 4-2. Bounding box
- 정확도 확인하기

- ‘chma’ 폴더 안에 ‘images_test’라는 임의의 폴더를 만들고 정확도를 확인할 이미지를 넣어준다. (본인은 훈련에 사용하지 않은 이미지를 넣어주었다.)

- 아래 코드 실행하면 content -> yolov5 -> runs -> detect -> exp에 객체 인식한 결과 이미지가 저장된다.
```python
import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets
!python detect.py --weights yolov5s.pt --source /content/drive/MyDrive/chma/export/test_image/경복궁_건숙문_처마_8.JPG
```

![image](https://user-images.githubusercontent.com/73512218/192141094-da791d38-ca5a-466a-a4e2-b901f6ce1178.png)

- 다음 코드 실행하여 결과를 눈으로 확인할 수 있다.

Image(filename='/content/yolov5/runs/detect/exp/경복궁_건숙문_처마_8.JPG')
(아래는 다른 예시 사진이다.)

![image](https://user-images.githubusercontent.com/73512218/192141109-b103962b-7c28-4121-ae83-6566f55573d7.png)
![image](https://user-images.githubusercontent.com/73512218/192141112-d1bfb94b-675b-40bb-befb-8e63b7af8b11.png)

> 4-3. 비교
- Polygon vs Bounding box

정확도 면에서는 Polygon이 대부분 80-99% 정도, Bounding box가 50-60% 정도였고
작업시간 면에서는 Polygon이 Bounding box에 비해 주석화 소요시간이 3배 이상 소요되었다. 많은 데이터를 바탕으로 작업을 한다고 가정하였을 때 소요시간의 격차는 더욱 커질 것이다. 따라서 본인의 시점에서는 Bounding box 방식을 채택한 후 훈련 과정에서 반복 횟수를 늘리거나 데이터 수의 확장 등 정확도를 높일 수 있는 방안을 마련하는 것이 더 효율적일것이라고 생각한다. 

### 5. 이미지 증강
- 증강을 시도한 계기 : 

받은 이미지 데이터의 개수가 한정적(ex. 건숙문 사진 60여개)이어서 더 높은 정확도를 얻어내기 위해서는 더 많은 이미지 데이터가 필요하다고 판단하여 증강을 통해 이미지 데이터의 수를 늘리고자 시도하였다.

- 이미지 증강 과정
  - Anaconda Navigator에서 Jupyter notebook을 킨다.
  
  ![image](https://user-images.githubusercontent.com/73512218/192141208-779991d6-79b1-40ef-b7ab-e1318b5d75c6.png)
  
  - Jupyter notebook 우측 상단에 New -> Python3 클릭
  
  ![image](https://user-images.githubusercontent.com/73512218/192141197-cb3ef66a-cdf3-43ac-94b1-e6d7f29ff029.png)

  - 첫 줄에서 Augmentor 라이브러리 설치한다.
  ```python
  pip install Augmentor
  ```
  
  - 다음 코드들을 실행하여 60개의 증강된 이미지를 얻는다.
  ```python
  import Augmentor
  # 증강 시킬 이미지 폴더 경로
  img = Augmentor.Pipeline("D:/이미지 증강")
  # 좌우 반전
  img.flip_left_right(probability=1.0) 
  # 증강 이미지 수
  img.sample(20)
  ```
  ![image](https://user-images.githubusercontent.com/73512218/192141326-a17df188-0646-4db3-b487-be501b7e999f.png)
  
  ```python
  import Augmentor
  # 증강 시킬 이미지 폴더 경로
  img = Augmentor.Pipeline("D:/이미지 증강")
  # 상하 반전
  img.flip_top_bottom(probability=1.0) 
  # 증강 이미지 수
  img.sample(20)
  ```
  ![image](https://user-images.githubusercontent.com/73512218/192141326-a17df188-0646-4db3-b487-be501b7e999f.png)
  
  ```python
  import Augmentor
  # 증강 시킬 이미지 폴더 경로
  img = Augmentor.Pipeline("D:/이미지 증강")
  # 왜곡
  img.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=8) 
  # 증강 이미지 수
  img.sample(20)
  ```
  ![image](https://user-images.githubusercontent.com/73512218/192141326-a17df188-0646-4db3-b487-be501b7e999f.png)

- 기존 58개 + 증강 이미지 60개의 이미지 데이터로 훈련 및 정확도 확인 결과 :
모두 같은 조건에서 58개로 훈련하였을 때와 118개로 훈련하였을 때 0.5에서 0.62로 증가.

![image](https://user-images.githubusercontent.com/73512218/192141466-e323eca2-1ea6-49ae-8370-41d3fa434820.png)
![image](https://user-images.githubusercontent.com/73512218/192141467-ad44a99f-2cc6-40aa-80c9-9c77cbee25c1.png)

***
