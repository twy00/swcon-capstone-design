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

<img width="300" alt="SG" src="https://github.com/twy00/swcon-capstone-design/blob/main/presentation/SG.png?raw=true"><br>


<img width="300" alt="SD" src="https://github.com/twy00/swcon-capstone-design/blob/main/presentation/SD.png?raw=true">


SG와 SD는 모두 FCSN이라 불리는 Fully Connected Sequence Network를 사용했는데, 이는 영상의 긴 프레임간의 관계를 저장하고 모델링하기 적합한 구조이다.

<img width="500" alt="FCSN" src="https://github.com/twy00/swcon-capstone-design/blob/main/presentation/FCSN.png?raw=true">

FCSN의 구조를 조금 더 구체적으로 살펴보면, 영상 프레임에서 feature를 추출하는 Encoder 부분과, 
이 feature를 사용해서 importance score를 매기고, 중요하다고 판단되는 프레임을 선택하는 decoder 부분으로 이루어져 있다. 
SG는 이 Encoder와 Decoder 전체 부분을 거쳐 영상에서 중요한 부분을 선택하고, SD는 Encoder 부분을 사용하여 영상이 생성된 것인지 Ground Truth 영상인지 판단한다.


<img width="600" alt="Framework" src="https://github.com/twy00/swcon-capstone-design/blob/main/presentation/Framework.png?raw=true"><br>

전반적인 모델 학습 구조는 위 그림과 같다.
먼저 원본 영상 프레임이 SG에 인풋으로 들어오면, 아웃풋으로 요약 영상을 생성한다.
SG가 아웃풋으로 생성한 요약 영상과 실제 Ground Truth 요약 영상이 SD에 인풋으로 들어가면, SD는 이 영상이 실제 요약 영상인지 생성된 것인지 판단한다.
SG와 SD의 아웃풋으로 각각 loss가 계산되고, 이를 통해 SG와 SD가 학습을 진행한다.

## Loss Functions
모델 학습을 위한 loss function은 총 세 개가 사용되었다.

<img width="400" alt="avdloss" src="https://github.com/twy00/swcon-capstone-design/blob/main/presentation/Advloss.png?raw=true">

첫 번째는 adversarial loss로, SG가 영상을 요약하고 SD가 구분을 할 때 발생하는 loss 값을 줄이는 것을 목적으로 한다.
SG는 영상을 실제 요약 영상처럼 만드는 방향으로, SD는 실제 영상과 생성된 영상을 잘 구분하기 위한 방향으로 학습한다.

<img width="400" alt="recloss" src="https://github.com/twy00/swcon-capstone-design/blob/main/presentation/Recloss.png?raw=true">


두 번째는 reconstruction loss로, 요약 영상을 만드는 SG가 decoder를 통해 재구현한 frame들과 실제 영상 frame간의 차이를 최소화하기 위한 방향으로 학습한다.
위 수식을 보면, 실제 영상의 프레임과 재구성된 영상의 프레임간 차이를 비교하고, 이를 최소화하는 방향으로 학습하는 것을 볼 수 있다.

<img width="400" alt="divloss" src="https://github.com/twy00/swcon-capstone-design/blob/main/presentation/Divloss.png?raw=true">

마지막으로 diversity loss는 SG를 거쳐 중요하다고 판단된 프레임들을 시각적으로 다양하게 만드는 방향으로 학습한다.
위 수식에서도 볼 수 있듯이, 추출된 모든 프레임간의 코사인 유사도를 각각 계산하고, 이를 최소화하는 방향으로 학습한다.

## Unpaired 데이터셋
이번 연구에서는 네 개의 데이터셋을 사용했으며, 그 종류는 TVSum, OVP, YouTube, SumMe이다.
전체 영상 수는 164개이고, 각 데이터셋에서 3개의 영상을 랜덤으로 뽑아 테스트셋으로 사용했다.

|    | Train | Test | Total | 
|:---:|:---:| :---:|:---:|
|TVSum| 47| 3 | 50 |
|OVP| 47| 3 | 50 |
|YouYube| 36 | 3 | 39 |
|SumMe| 22 | 3 | 25 |
|__total__| __152__ | __12__ | __164__ |

이번 프로젝트에서는 unpaired 데이터셋을 만들어서 학습을 진행하려 했기 때문에, 트레인셋으로 사용할 152개 데이터를 랜덤으로 76개씩 나누어서 하나는 SG의 인풋으로 원본 영상만 사용하고, 다른 하나는 SD의 Ground Truth 인풋으로 요약된 영상만을 사용했다. 이런 방법으로 unpaired 데이터셋을 생성하였고, 이를 통해 영상 요약 모델이 훈련 데이터에 종속되는 것을 줄이고자 했다.

## Feature Extraction

<img width="300" alt="feature" src="https://github.com/twy00/swcon-capstone-design/blob/main/presentation/Feature.png?raw=true">


모델에 실제 인풋으로 들어갈 데이터셋을 만들기 위한 feature creation을 진행했다.
먼저 수천장의 프레임으로 이루어진 하나의 영상 데이터가 ImageNet 데이터로 pre train된 GoogLeNet을 거쳐 나온다. 
GoogLeNet의 pool5 layer라 불리는 부분에서 feature를 추출한다.
그러면 한 프레임 당 1024 차원의 feature가 생성되는데, 이 feature를 각 15프레임마다 하나씩 추출하여 다운샘플링을 했다.

## 결과

<img width="300" alt="f score equation" src="https://github.com/twy00/swcon-capstone-design/blob/main/presentation/F-score-equation.png?raw=true">


모델 성능 측정을 위해 위 그림과 같은 방법으로 F1 score를 측정했다.

학습 결과는 아래 표와 같이 0.5296의 F1 score가 측정되었다.
|    | F1 score |
|:---:|:---:|
|__이번 연구__| __0.5296__ |
|FCSN <br>(SumMe로 테스트)| 0.448 | 
|FCSN <br>(TVSum으로 테스트)| 0.536 | 
|FCSN-advloss <br> (SumMe로 테스트)| 0.465 |
|FCSN-advloss <br> (TVSum으로 테스트)| 0.553 |


아래 그래프는 모델 학습 중 10 epoch 당 F1 score를 나타낸 것이다.

<img width="300" alt="f score" src="https://github.com/twy00/swcon-capstone-design/blob/main/presentation/F-score.png?raw=true">


## 결론

* 캡스톤디자인 프로젝트에서는 Unpaired 데이터셋과 Adversarial 방법을 사용하여, 영상 요약 모델 학습이 데이터에 종속되는 것을 줄이고자 했다.

* 이 프로젝트를 진행하면서 부족했다고 생각한 것은, 결과를 paired 데이터셋을 사용했을 때와의 성능을 비교하지 않았다는 점이다. 따라서 추후 연구로 이 모델을  paired 데이터셋으로 학습한 후 성능을 비교하는 과정이 필요한 것으로 보인다.

* 이번 프로젝트의 다른 한계점중 하나로 영상 요약 모델을 FCSN만 사용했다는 점을 보완하기 위해, LSTM 등 영상 요약 성능이 검증된 모델을 사용하여 학습을 하고 결과를 비교하는 것을 추후 연구로 진행할 계획이다.
