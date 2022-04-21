# Image Classification on Amazon Sagemaker Studio

![machine_learning_with_sagemaker](machine_learning_with_sagemaker.png)

Amazon sagemaker는 머신러닝 워크플로우를 위한 통합적인 도구입니다. 다시 말해서 data collection을 제외한 전 영역을 지원합니다.

## 목적

공공데이터를 이용해서 data preprocessing, computer vision modeling, and model deployment를 Amazon SageMaker Studio에서 실제로 구현하는 법을 배울 수 있습니다.


## Amazon Sagemaker Studio의 4가지 모델링 방법

![sagemaker_modeling_type.png](sagemaker_modeling_type.png)

Amazon sagemaker는 모델 훈련을 위한 4가지 방법을 지원합니다. 이 예제에서는 가장 수요가 많은 두번째 방법(Custom script on supported framework)으로 AWS 환경에서 모델을 훈련하는 방법을 다룹니다.

두번째 방법은 장점이 많이 있습니다. AWS에서 지원하는 프레임워크이기 때문에 인스턴스 최적화가 잘 되어 있고 라이브러리 설치와 같은 환경에 신경쓰지 않고 모델링에 집중할 수 있습니다. 그리고 ```training job```에서는 모델학습이 완료되면 사용된 인스턴스를 자동으로 종료해주기 때문에 과금에 대한 부담도 줄일 수 있습니다.