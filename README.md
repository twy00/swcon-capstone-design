# Unpaired 데이터셋을 사용한 Adversarial Networks 기반 영상 요약 모델 개발 및 평가
## 2021-1 소프트웨어융합 캡스톤디자인 최종발표
* [소프트웨어융합학과 유태원](https://twy00.github.io)
* [최종 발표자료](https://github.com/twy00/swcon-capstone-design/blob/main/presentation/final_presentation.pdf)

## 개요
### 영상 요약
* 지난 수 년간, 온라인에 많은 양의 영상이 업로드 되면서, 사용자들이 영상을 효율적으로 찾을 수 있도록 하거나 분석의 용이성을 위해 영상을 요약하여 보여주는 시스템이 요구되고 있다.
* 영상 요약 방법에는 Key Frame Selection(프레임 선택)과 Key Shots Selection(구간 선택)이 있는데, 이번 연구에서는 Key Frame Selection방법을 사용하여 영상을 요약하는 모델을 개발하고 평가한다.
### Unpaired 데이터셋 & Adversarial Networks
* 지도학습을 사용한 영상 요약 모델 학습은 데이터셋을 생성하는데 큰 비용과 시간이 든다는 문제가 있다. 또한 제공되는 데이터셋 영상의 개수와 분야가 매우 한정적이어서 모델의 데이터 종속성이 커진다.
* 이 문제를 해결하기 위해, Unpaired 데이터셋과 Adversarial Networks 구조를 사용하여 모델이 영상의 분야에 종속되는 한계점을 해결하고자 했다.

## 모델 구조
모델은 크게 Summary Generator(SG)와 Summary Discriminator(SD)가 있다.
SG는 영상의 전체 프레임이 인풋으로 주어지면, 각 프레임의 importance score를 매기고, 이를 통해 영상의 요약 부분이 생성된다. 
SD는 실제 Ground Truth 요약 영상과 SG가 생성한 요약 영상을 인풋으로 받아, 이 영상이 실제로 요약된 영상인지, 아니면 SG가 생성한 영상인지 구분하는 모델이다.
![Summary Generator(SG) 구조 ](https://github.com/twy00/swcon-capstone-design/blob/main/presentation/SG.png)
![Summary Discriminator(SD) 구조 ](https://github.com/twy00/swcon-capstone-design/blob/main/presentation/SG.png)

SG와 SD는 모두 FCSN이라 불리는 Fully Connected Sequence Network를 사용했는데, 이는 영상의 긴 프레임간의 관계를 저장하고 모델링하기 적합한 구조이다.

![Fully Connected Sequence Network(FCSN) 구조 ](https://github.com/twy00/swcon-capstone-design/blob/main/presentation/FCSN.png)

FCSN의 구조를 조금 더 구체적으로 살펴보면, 영상 프레임에서 feature를 추출하는 Encoder 부분과, 
이 feature를 사용해서 importance score를 매기고, 중요하다고 판단되는 프레임을 선택하는 decoder 부분으로 이루어져 있다. 
SG는 이 Encoder와 Decoder 전체 부분을 거쳐 영상에서 중요한 부분을 선택하고, SD는 Encoder 부분을 사용하여 영상이 생성된 것인지 Ground Truth 영상인지 판단한다.

전반적인 모델 학습 구조는 아래 그림과 같다.
![모델 학습 구조](https://github.com/twy00/swcon-capstone-design/blob/main/presentation/Framework.png)

먼저 원본 영상 프레임이 SG에 인풋으로 들어오면, 아웃풋으로 요약 영상을 생성한다.
SG가 아웃풋으로 생성한 요약 영상과 실제 Ground Truth 요약 영상이 SD에 인풋으로 들어가면, SD는 이 영상이 실제 요약 영상인지 생성된 것인지 판단한다.
SG와 SD의 아웃풋으로 각각 loss가 계산되고, 이를 통해 SG와 SD가 학습을 진행한다.

## Loss Functions
모델 학습을 위한 loss function은 총 세 개가 사용되었다.

