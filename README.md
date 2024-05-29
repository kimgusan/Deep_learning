# Deep Learning

-   인공 신경망(Artivicial Neural Network)의 층을 연속적으로 깊게 쌓아올려 데이터를 학습하는 방식을 의미한다.
-   인간이 학습하고 기억하는 매커니즘을 모방한 기계학습이다.
-   인간은 학습 시, 뇌에 있는 뉴런이 자극을 받아들여서 일정 자극 이상이 되면, 화학물질을 통해 다른 뉴런과 연결되며 해당 부분이 발달한다.
-   자극이 약하거나 기준치를 넘지 못하면, 뉴런은 연결되지 않는다.
-   입력한 데이터가 활성 함수에서 임계점을 넘게 되면 출력된다.
-   초기 인공 신경망(Perceptron)에서 깊게 층을 쌓아 학습하는 딥 러닝으로 발전한다.
-   딥 러닝은 Input nodes layer, Hidden nodes layer, Output nodes layer, 이렇게 세 가지 층이 존재한다.

## 목차

1. [Perceptron](#perceptron)
2. [Tensorflow](#tensorflow)
3. [CNN](#cnn)

---

## Perceptron

<div id="Perceptron">

### SLP (Single Layer Perceptron), 단층 퍼셉트론, 단일 퍼셉트론

-   가장 단순한 형태의 신경망으로서, Hidden Layer가 없고 Single Layer로 구성되어 있다.
-   퍼셉트론의 구조는 입력 feature와 가중치, activation function, 출력 값으로 구성되어 있다.
-   신경 세포에서 신호를 전달하는 축삭돌기의 역할을 퍼셉트론에서는 가중치가 대신하고,  
     입력 값과 가중치 값은 모두 인공 뉴련(활성 함수)으로 도착한다.
-   가중치의 값이 클수록 해당 입력 값이 중요하다는 뜻이고, 인공 뉴런(활성 함수)에 도착한 각 입력 값과 가중치 값을 곱한 뒤 전체 합한 값을 구한다.
-   인공 뉴런(활성 함수)은 보통 시그모이드 함수와 같은 계단 함수를 사용하여,  
     합한 값을 확률로 변환하고 이 때, 임계치를 기준으로 0 또는 1을 출력한다.

-   로지스틱 회귀 모델이 인공 신경망에서는 하나의 인공 뉴런으로 볼 수 있다.
-   결과적으로 퍼셉트론의 회귀 모델과 마찬가지로 실제 값과 예측 값의 차이가 최소가 되는 가중치 값을 찾는 과정이 퍼셉트론이 학습하는 과정이다.
-   최소 가중치 값을 설정한 뒤 입력 feature 값으로 예측 값을 계산하고, 실제 값과의 차이를 구한 뒤 이를 줄일 수 있도록 가중치 값을 변경한다.
-   퍼셉트론의 활정화 정도를 편향(bias)으로 조절할 수 있으며, 편향을 통해 어느정도의 자극을 미리 주고 시작 할 수 있다.
-   뉴런이 활성화되기 위해 필요한 자극이 1000이라고 가정하면, 입력 값을 500만 받아도 편향을 2로 주어 1000을 만들 수 있다.

-   퍼셉트론의 출력 값과 실제 값의 차이를 줄여나가는 방향성으로 계속해서 가중치 값을 변경하며, 이 때 경사하강법을 사용한다.

#### SGD (Stochastiic Gradient Descent), 확률적 경사 하강법

-   경사 하강법 방식은 전체 학습 데이터를 기반으로 계산한다. 하지만 입력 데이터가 크고 레이어가 많을 수록 많은 자원이 소모된다.
-   일반적으로 메모리 부족으로 인해 연산이 불가능하기 때문에, 이를 극복하기 위해 SGD 방식이 도입되었다.
-   전체 학습 데이터 중, 단 한 건만 임의로 선택하여 경사 하강법을 실시하는 방식을 의미한다.
-   많은 건 수 중에 한 건만 실시하기 때문에, 빠르게 최적점을 찾을 수 있지만 노이즈가 심하다.
-   무작위로 추출된 샘플 데이터에 대해 경사 하강법을 실시하기 때문에 진폭이 크고 불안정해 보일 수 있다.
-   일반적으로 사용하지 않고, SGD를 얘기할 때에는 보통 미니 배치 경사 하강법을 의미한다.

-   전체 학습 데이터 중, 특정 크기(Batch 크기)만큼 임의로 선택해서 경사 하강법을 실시한다. 이 또한, 확률적 경사 하강법

-   전체 학습 데이터가 1000건이라고 하고, batch size를 100건이라 가정하면, 전체 데이터를 batch size만큼 나눠서 가져온 뒤 섞고, 경사하강법을 계산한다.  
    이 경우, 10번 반복해야 1000개의 데이터가 모두 학습되고 이를 epoch라고 한다. 즉, 10 epoch \* 100 batch 이다.

### (MLP) Multi Layer Perceptron, 다층 퍼셉트론, 다중 퍼셉트론

-   보다 복잡한 문제의 해결을 위해서 입력층과 출력층 사이에 은닉층이 포함되어 있다.
-   퍼셉트론을 여러층 쌓은 인공 신경망으로서, 각 층에는 활성함수를 통해 입력을 처리한다.
-   층이 깊어질 수록 정확한 분류가 가능해지지만, 너무 깊어지면 Overfitting이 발생한다.

#### ANN (Artificial Neural Network), 인공 신경망

-   은닉층이 1개일 경우 이를 인공 신경망이라고 한다.

#### DNN (Deep Neural Network), 심층 신경망

-   은닉층이 2개 이상일 경우 이를 심층 신경망이라고 한다.

#### **Back-propagation, 역전파**

-   심층 신경망에서 최종 출력(예측)을 하기 위한 식이 생기지만 식이 너무 복잡해지기 때문에 편미분을 진행하기에 한계가 있다.
-   즉, 편미분을 통해 가중치 값을 구하고, 경사 하강법을 통해 가중치 값을 업데이트 하며, 손실 함수의 최소값을 찾아야 하는데, 순방향으로는 복잡한 미분식을 계산할 수가 없다.  
    따라서 미분의 연쇄 법칙(Chain Rule)을 사용하여 역방향으로 편미분을 진행한다.

### Activation Function, 활성화 함수

-   인공 신경망에서 입력 값에 가중치를 곱한 뒤 합한 결과를 적용하는 함수이다.

---

1. **시그모이드 함수**
    - 은닉층이 아닌 최종 활성화 함수, 출력층에서 사용된다.
    - 은닉층에서 사용 시, 입력 값이 양의 방향으로 큰 값일 경우 출력값의 변화가 없으며, 음의 방향도 마찬가지이다.  
      평균이 0이 아니기 때문에 정규 분포 형태가 아니고, 이는 방향에 따라 기울기가 달려져서 탐색 경로가 비효율적(지그재그)이 된다.
2. **소프트맥스 함수**

    - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
    - 시그모이드와 유사하게 0 ~ 1 사이의 값을 출력하지만, 이진 분류가 아닌 **다중 분류**를 통해 모든 확률값이 1이 되도록 해준다.
    - 여러 개의 타겟 데이터를 분류하는 다중 분류의 최종 활성화 함수(출력층)로 사용된다.

3. 탄젠트 함수

    - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
    - 은닉층에서 사용 시, 시그모이드와 달리 -1 ~ 1 사이의 값이 출력해서 평균이 0이 될 수 있지만,  
      여전히 입력 값의 양의 밥향으로 큰 값일 경우 출력값의 변화가 미비하고 음의 방향도 마찬가지 이다.

4. **렐루 함수**
    - 대표적인 은닉층의 활성 함수이다.
    - 입력 값이 0보다 작으면 출력은 0, 0보다 크면 입력값을 출력하게 된다.

### Optimizer, 최적화

-   최적의 경사 하강법을 적용하기 위해 필요하며, 최소값을 찾아가는 방법들을 의미한다.
-   loss를 줄이는 방향으로 최소 loss를 보다 빠르고 안정적으로 수렴할 수 있어야 한다.

#### Momentum

-   가중치를 계속 업데이트할 때마다 이전의 값을 일정 수준 반영시키면서 새로운 가중치로 업데이트한다.
-   지역 최소값에서 벗어나지 못하는 문제를 해결할 수 있으며, 진행했던 방향만큼 추가적으로 더하여, 관성처럼 빠져나올 수 있게 해준다.

#### AdaGrad (Adaptive Gradient)

-   가중치 별로 서로 다른 학습률을 동적으로 적용한다.
-   적게 변화된 가중치는 보다 큰 학습률을 적용하고, 많이 변화된 가중치는 보다 작은 학습률을 적용시킨다.
-   처음에는 큰 보폭으로 이동하다가 최소값에 가까워질 수록 작은 보폭으로 이동하게 된다.
-   과거의 모든 기울기를 사용하기 때문에 학습률이 급격히 감소하여, 분모가 커짐으로써 학습률이 0에 가까워지는 문제가 있다.

#### RMSProp (Root Mean Sqaure Propagation)

-   AdaGrad의 단점을 보완한 기법으로서, 학습률이 지나치게 작아지는 것을 막기 위해 지수 가중 평균법(exponentially weighted average)을 통해 구한다.
-   지수 가중 평균법이란, 데이터의 이동 평균을 구할 때 오래된 데이터가 미치는 영향을 지수적으로 감쇠하도록 하는 방법이다.
-   이전의 기울기들을 똑같이 더해가는 것이 아니라 훨씬 이전의 기울기는 조금 반영하고 최근의 기울기를 많이 반영한다.
-   feature마다 적절한 학습률을 적용하여 효율적인 학습을 진행할 수 있고, AdaGrad보다 학습을 오래 할 수 있다.

#### Adam (Adaptive Moment Estimation)

-   Momentum과 RMSProp 두 가지 방식을 결합한 형태로서, 진행하던 속도에 관성을 주고, 지수 가중 평균법을 적용한 알고리즘이다.
-   최적화 방법 중에서 가장 많이 사용되는 알고리즘이며, 수식은 아래와 같다.

</div>

## tensorflow

<div id="tensorflow">

### Tensorflow, 텐서플로우

-   구글이 개발한 오픈소스 소프트웨어 라이브러리이며, 머신러닝과 딥러닝을 쉽게 사용할 수 있도록 다양한 기능을 제공한다.
-   주로 이미지 인식이나 반복 신경망 구성, 기계 번역, 필기 숫자 판별 등을 위한 각종 신경망 학습에 사용된다.
-   딥러닝 모델을 만들 때, 기초부터 세세하게 작업해야 하기 때문에 진입장벽이 높다.

### Keras, 케라스

-   일반 사용 사례에 최적화되고 "최적화, 간단, 일관, 단순화"된 인터페이스를 제공한다.
-   손쉽게 딥러닝 모델을 개발하고 활용할 수 있도록 직관적인 API를 제공한다.
-   텐서플로우 2버전 이상부터 케라스가 포함되었기 때문에 텐서플로우를 통해 케라스를 사용한다.
-   기존 Keras 패키지보다는 이제 Tensorflow에 내장된 Keras 사용이 더 권장된다.

---

#### Sequential API

-   간단한 모델을 구현하기에 적합하고 단순하게 층을 쌓는 방식으로 쉽고 사용하기가 간단하다.
-   단일 입력 및 출력만 있으므로 레이어를 공유하거나 여러 입력 또는 출력을 가질 수 있는 모델을 생성할 수 없다.

#### Funcional API

-   Funtional API는 Sequential API로는 구현하기 어려운 복잡한 모델들을 구현할 수 있다.
-   여러 개의 입력 및 출력을 가진 모델을 구현하거나 층 간의 연결 및 연산을 수행하는 모델 구현 시 사용한다.

---

### Grayscale, RGB

-   흑백 이미지와 컬러 이미지는 각 2차원과 3차원으로 표현될 수 있다.
-   흑백 이미지는 0 ~ 255를 갖는 2차원 배열(높이 X 너비)이고,  
    컬러 이미지는 0 ~ 255를 갖는 R, G, B 2차원 배열 3개를 갖는 3차원(높이 X 너비 X 채널)이다.

### Grayscale Image Matrix

-   검은색에 가까운 색은 0에 가깝고 흰색에 가까우면 255에 가깝다.
-   모든 픽셀이 feature이다.

---

### Callback API (활용성이 높음!)

-   모델이 학습 중에 충돌이 발생하거나 네트워크가 끊기면, 모든 훈련 시간이 낭비될 수 있고,  
    과적합을 방지하기 위해 훈련을 중간에 중지해야 할 수도 있다.
-   모델이 학습을 시작하면 학습이 완료될 때까지 아무런 제어를 하지 못하게 되고,  
    신경망 훈련을 완료하는 데에는 몇 시간 또는 며칠이 걸릴 수 있기 때문에 모델을 모니터링하고 제어할 수 있는 기능이 필요하다.
-   훈련 시(fit()) Callback API를 등록시키면 반복 내에서 특정 이벤트 발생마다 등록된 callback이 호출되어 수행된다.

**1) ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weight_only=False, mode='auto')**

-   특정 조건에 따라서 모델 또는 가중치를 파일로 저장한다.
-   filepath: "weight.{epoch: 03d}-{val_loss:.4f}-{acc:.4f}.weights.hdf5" 와 같이 모델의 체크포인트를 저장한다.
-   monitor: 모니터링할 성능 지표를 작성한다.
-   save_best_only: 가장 좋은 성능을 나타내는 모델을 저장할 지에 대한 여부
-   save_weights_only: weights만 저장할 지에 대한 여부
-   mode: {auto, min, max} 중 한 가지를 작성한다. monitor의 성능 지표에 따라 좋은 경우를 선택한다.  
    \*monitor의 성능 지표가 감소해야 좋은 경우 min, 증가해야 좋은 경우 max, monitor의 이름으로부터 자동으로 유추하고 싶다면 auto를 사용한다.

**2) ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto, min_lr=0')** (LR: Learning Rate)

-   특정 반복동안 성능이 개선되지 않을 때, 학습률을 동적으로 감소시킨다.
-   monitor: 모니터링할 성능 지표를 작성한다.
-   factor: 학습률을 감소시킬 비율, 새로운 학습률 = 기존 학습률 \* factor
-   patience: 학습률을 줄이기 전에 monitor할 반복 횟수
-   mode: {auto, min, max} 중 한 가지를 작성한다. monitor의 성능 지표에 따라 좋은 경우를 선택한다.  
    \*monitor의 성능 지표가 감소해야 좋은 경우 min, 증가해야 좋은 경우 max, monitor의 이름으로부터 자동으로 유추하고 싶다면 auto를 사용한다.

**3) EarlyStopping(monitor='val_loss'm patient=0, verbose=0, mode='auto')**

-   특정 반복동안 성능이 개선되지 않을 때, 학습을 조기에 중단한다.
-   monitor: 모니터링할 성능 지표를 작성한다.
-   patience: Early Stopping을 적용하기 전에 monitor할 반복 횟수.
-   mode: {auto, min, max} 중 한 가지를 작성한다. monitor의 성능 지표에 따라 좋은 경우를 선택한다.  
    \*monitor의 성능 지표가 감소해야 좋은 경우 min, 증가해야 좋은 경우 max, monitor의 이름으로부터 자동으로 유추하고 싶다면 auto를 사용한다.

#### <div id="tensowflow-code">tensorflow Code</div>

<details>
    <summary> 1. keras에서 불러온 이미지에 대하여 이미지 표기 함수 코드</summary>

        def show_images(images, targets, ncols=8):
        figure, axs = plt.subplots(figsize=(22, 6), nrows=1, ncols=ncols)
        for i in range(ncols):
            axs[i].imshow(images[i], cmap='gray')
            axs[i].set_title(class_names[targets[i]])

        show_images(train_images[:8], train_targets[:8])
        show_images(train_images[8:16], train_targets[8:16])

</details>

<details>
    <summary> 2. Sequential API Code</summary>

        from tensorflow.keras.layers import Dense, Flatten
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.losses import CategoricalCrossentropy
        from tensorflow.keras.optimizers import Adam

        # shape
        INPUT_SIZE = 28

        model = Sequential([
            # 입력층
            Flatten(input_shape=(INPUT_SIZE, INPUT_SIZE)),
            # 은닉층
            Dense(64, activation='relu'),
            # 은닉층
            Dense(128, activation='relu'),
            # 출력층 (다중 분류 이기 때문에 활성 함수는 softmax 사용)
            Dense(10, activation='softmax')
        ])

        # # 경사하강법 optimizer, 및 최적화
        # 손실함수는 다중 함수이기 때문에 CategoricalCrossentropy 사용
        model.compile(optimizer=Adam(0.001), loss=CategoricalCrossentropy(), metrics=['acc'])

</details>

<details>
    <summary>3. 검증데이터를 포함한 정확도 그래프 표현</summary>
        #라이브러리 호출
        import matplotlib.pyplot as plt

        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='validation')
        plt.legend()
        plt.show()

</details>

<details>
<summary>4. 검증 데이터 정확도 및 손실 함수 확인</summary>

    # 검증데이터를 확인하기 위한 evaluate 함수 사용
    model.evaluate(test_images, test_oh_targets, batch_size=32)

</details>

<details>
    <summary> 5. Funtional API Code</summary>

        ### call 매직 메소드
        # call 함수 (매직 메소드)
        class Test:
            def __call__(self,data):
                return data + 10

---

        # call 함수 덕분에 생성자 뒤에 값을 넣어줘서 사용이 가능하다.

        from tensorflow.keras.layers import Layer, Input, Dense, Flatten
        from tensorflow.keras.models import Model

        INPUT_SIZE = 28

        def create_model():
            input_tensor = Input((INPUT_SIZE,INPUT_SIZE))
            x = Flatten()(input_tensor)
            x = Dense(64, activation='relu')(x)
            x = Dense(128, activation='relu')(x)
            output = Dense(10, activation='softmax')(x)

            model = Model(inputs=input_tensor, outputs=output)
            return model


        model = create_model()
        model.summary()

</details>

<details>
    <summary>6. tensorflow 전처리 과정 (array객체 변환, 원-핫 인코딩, 훈련/검증/테스트 데이터 분리)</summary>

    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    import numpy as np

    (train_images, train_targets), (test_images, test_targets) = fashion_mnist.load_data()

    # array 객체 변환 및 실수 변환, 색상을 표현하기 위해 255.0 으로 변환
    def get_preprocessed_data(images, targets):
        images = np.array(images / 255.0, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return images, targets

    # 타겟 데이터에 대하여 원-핫 인코딩 메소드 생성
    def get_preprocessed_ohe(images, targets):
        images, targets = get_preprocessed_data(images, targets)
        oh_targets = to_categorical(targets)

        return images, oh_targets

    # 훈련, 검증, 테스트 데이터로 분리하기 위한 메소드 생성
    def get_train_valid_test(train_images, train_targets, test_images, test_targets, validation_size=0.2, random_state=124):
        train_images, train_oh_targets = get_preprocessed_ohe(train_images, train_targets)
        test_images, test_oh_targets = get_preprocessed_ohe(test_images, test_targets)

        train_images, validation_images, train_oh_targets, validation_oh_targets = \
        train_test_split(train_images, train_oh_targets, stratify=train_oh_targets, test_size=validation_size, random_state=random_state)

        return (train_images, train_oh_targets), (validation_images, validation_oh_targets), (test_images, test_oh_targets)

---

    from tensorflow.keras.datasets import fashion_mnist

    (train_images, train_targets), (test_images, test_targets) = fashion_mnist.load_data()

    (train_images, train_oh_targets), (validation_images, validation_oh_targets), (test_images, test_oh_targets) = \
    get_train_valid_test(train_images, train_targets, test_images, test_targets)

    print(train_images.shape, train_oh_targets.shape)
    print(validation_images.shape, validation_oh_targets.shape)
    print(test_images.shape, test_oh_targets.shape)

</details>

<details>
    <summary>tip. model.predict (pred_prob), np.argmax</summary>

        # 훈련과 정답의 차원을 맞추기 위해 차원을 늘리는 작업
        import numpy as np
        np.expand_dims(test_images[0], axis=0).shape

        # 정답이 나올 확률
        pred_prob = model.predict(np.expand_dims(test_images[8500], axis=0))
        print(pred_prob)

        # 정답이 나올 확률 및 정담을 출력
        pred_proba = model.predict(np.expand_dims(test_images[326], axis=0))
        print('softmax output:', pred_proba)

        # argmax() : 가장 높은 값의 인덱스를 찾아서 표기하는 함수
        pred = np.argmax(np.squeeze(pred_proba))
        print('predicted target value:', pred)

</details>

<details>
    <summary>7. Callback API(ModelCheckpoint, ReduceLROnPlateau, EarlyStopping)</summary>

        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.losses import CategoricalCrossentropy
        from tensorflow.keras.callbacks import ModelCheckpoint

        model = create_model()
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['acc'])

        mcp_cb = ModelCheckpoint(
            filepath="./callback_files/weights.{epoch:03d}-{val_loss:.4f}-{acc:.4f}.weights.h5",
            monitor='val_loss',
            save_best_only=False,
            # Model이 아는 weight 를 저장할 때 True설정
            save_weights_only=True,
            mode='min'
        )

        rlr_cb = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=2,
            mode='min'
        )

        ely_cb = EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min'
        )

        history = model.fit(x=train_images, y=train_oh_targets, validation_data=(validation_images, validation_oh_targets), batch_size=64, epochs=20, callbacks=[mcp_cb, rlr_cb, ely_cb])

</details>

<details>
    <summary>8. callback을 이용한 가중치 파일 불러오기</summary>
        model.load_weights('./callback_files/')
        model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['acc'])
</details>

</div>

<hr>

## CNN

<div id="cnn">

### CNN (Convolutional Neural Network), 합성곱 신경망

-   실제 이미지 데이터는 분류 대상이 이미지에서 고정된 위치에 있지 않은 경우가 대부분이다.
-   실제 이미지 데이터를 분류하기 위해서는, 이미지의 각 feature들을 그대로 학습하는 것이 아닌, CNN으로 패턴을 인식한 뒤 학습해야 한다.

-   이미지의 크기가 커질 수록 굉장히 많은 Weight가 필요하기 때문에 분류기에 바로 넣지 않고, 이를 사전에 추출 및 축소해야 한다.
-   CNN은 인간의 시신경 구조를 모방한 기술로서, 이미지의 패턴을 찾을 때 사용한다.
-   Feature Extraction을 통해 각 단계를 거치면서, 함축된 이미지 조각으로 분리되고 각 이미지 조각을 통해 이미지의 패턴을 인식한다.

-   CNN은 분류하기에 적합한 최적의 feature를 추출하고, 최적의 feature를 추출하기 위한 최적의 Weight와 filter를 계산한다.

#### Filter

-   일반적으로 정방 행렬로 구현되어 있고, 원본 이미지에 슬라이딩 윈도우 알고리즘을 사용하여 순차적으로 새로운 픽셀값을 만들면서 적용한다.
-   사용자가 목적에 맞는 특정 필터를 만들거나 기존에 설계된 다양한 필터를 선택하여 이미지에 적용한다.  
    하지만, CNN은 최적의 필터값을 학습하여 스스로 최적화 한다.

#### Kernel

-   filter 안에 1 ~ n개의 커널이 존재한다. 커널의 개수는 반드시 이미지의 채널 수와 동일해야 한다.
-   kernel Size는 가로 X 세로를 의미하며, 가로와 세로는 서로 다를 수 있지만 보통은 일치시킨다.
-   kernel Size가 크면 클 수록 입력 이미지에서 더 많은 feature 정보를 가져올 수 있지만, 큰 사이즈의 kernel로 Convolution Backbone을 할 경우 훨씬 더 많은 연산량과 파라미터가 필요하다.

\*\* 커널, 채널, 필터

#### Stride

-   입력 이미지에 Convolution Filter를 적용할 때 Slide Window가 이동하는 간격을 의미한다.
-   기본 stride는 1이지만, 2를 적용하면 입력 feature map 대비 출력 feature map의 크기가 절반정도 줄어든다.
-   stride를 키우면 feature 정보를 손실할 가능성이 높아지지만, 오히려 불필요한 특성을 제거하는 효과를 가져올 수 있고 Convolution 연산 속도를 향상 시킨다.

#### Padding

-   Filter를 적용하여 Convolution 수행 시 출력 feature map이 입력 feature map 대비 계속해서 작아지는 것을 막기 위해 사용한다.
-   Filter 적용 전, 입력 feature map의 상하좌우 끝에 각각 열과 행을 추가한 뒤, 0으로 채워서 크기를 증가시킨다.
-   출력 이미지의 크기를 입력 이미지의 크기와 동일하게 유지하기 위해서 직접 계산할 필요 없이 "same"이라는 값을 통해 입력 이미지의 크기와 동일하게 맞출 수 있다.

#### Pooling

-   Convolutoin이 적용된 feature map의 일정 역영별로 하나의 값을 추출하여 feature map의 사이즈를 줄인다.
-   보통은 Convolution -> Relu activation -> Pooling 순서로 적용한다.
-   비슷한 feature들이 서로 다른 이미지에서 위치가 달라지면서 다르게 해석되는 현상을 중화시킬 수 있고,
    feature map의 크기가 줄어들기 때문에, 연산 성능이 향상된다.
-   Max Pooling과 Average Pooling이 있으며, Max Pooling은 중요도가 가장 높은 feature를 추출하고, Average Pooling은 전체를 버무려서 추출한다.

#### 🚩 정리

-   Stride를 증가시키는 것과 Pooling을 적용하는 것을 출력 feature map의 크기를 줄이는데 사용하는 것이다.
-   Convolution 연산을 진행하면서, feature map의 크기를 줄이면, 위치 변화에 따른 feature의 영향도도 줄어들기 때문에 과적합을 방지할 수 있는 장점이 있다.
-   Pooling의 경우 특정 위치의 feature 값이 손실되는 이슈 등으로 인하여 최근 Advanced CNN에서는 많이 사용되지 않는다.
-   Classifier에서는 Fully Connected Layer의 지나친 연결로 인해 많은 파라미터가 생성되므로 오히러 과적합이 발생할 수 있다.

-   위의 상황을 대비하기 위해 Dropout을 사용해서 Layer간 연결을 줄일 수 있으며 과적합을 방지할 수 있다. (뉴런을 비활성화 시키는 작업.)

---

### CNN (RGB)

-   RGB 영상이기 때문에 필터의 경우 '3'이 적용된다.
-   input data 와의 차원을 맞추기 위해 Squeeze 를 사용한다 (<-> Unsqueeze)

---

### CNN Performance

-   CNN 모델을 제작할 때, 다양한 기법을 통해 성능 개선 및 과적합 계산이 가능하다.

#### Weight Initialization, 가중치 초기화

-   처음 가중치를 어떻게 줄 것인지를 정하는 방법이며, 처음 가중치를 어떻게 설정하느냐에 따라 모델의 성능이 크게 달라질 수 있다.

> 1. 사비에르 글로로트 초기화
>
> -   고정된 표준편차를 사용하지 않고, 이전 층의 노드 수에 맞게 현재 층의 가중치를 초기화한다.
> -   층마다 노드 개수를 다르게 설정하더라도 이에 맞게 가중치가 초기화되기 때문에 고정된 표준편차를 사용하는 것보다 이상치에 민감하지 않다.
> -   활성화 함수가 ReLU일 때, 층이 지날 수록 활성화 값이 고르지 못하게 되는 문제가 생겨서, **출력층에서만 사용**한다.

> 2. 카이밍 히 초기화
>
> -   고정된 표준편차를 사용하지 않고, 이전 층의 노드 수에 맞게 현재 층의 가중치를 초기화한다.
> -   층마다 노드 개수를 다르게 설정하더라도 이에 맞게 가중치가 초기화되기 때문에 고정된 표준편차를 사용하는 것보다 이상치에 민감하지 않다.
> -   활성화 함수가 ReLU일 때, 추천하는 초기화 방법으로서, 층이 깊어지더라도 모든 활성값이 고르게 분포된다.

#### Batch Normalization, 배치 정규화

-   입력 데이터 간에 값의 차이가 발생하면, 가중치의 비중도 달라지기 때문에 층을 통과할 수록 편차가 심해진다.  
    이를 내부 공변량 이동(Internel Convariant Shift)이라고 한다.
-   가중치의 값의 미중이 달라지면, 특정 가중치에 중점을 두면서 경사 하강법이 진행되기 때문에,  
    모든 입력값을 표준 정규화하여 최적의 parameter를 보다 빠르게 학습할 수 있도록 해야한다.
-   가중치를 초기화할 때 민감도를 감소시키고, 학습 속도를 증가시키며, 모델을 일반화하기 위해서 사용한다.

-   BN은 activation function 앞에 적용하면, weight 값은 평균이 0, 분산이 1인 상태로 정규분포가 된다.
-   ReLU가 activation으로 적용되면 음수에 해당하는(절반 정도) 부분이 0이 된다.  
    이러한 문제를 해결하기 위해서 (감마)와 (베타)를 사용해서 음수부분이 모두 0이 되는 것을 막아준다.

#### Batch Size

-   batch size를 작게 하면, 적절한 noise가 생겨서 overfitting을 방지하게 되고, 모델의 성능을 향상시키는 계기가 될 수 있지만, 너무 작아서는 안된다.
-   batch size를 너무 작게 하는 경우에는 batch당 sample 수가 작아져서 훈련 데이터를 학습하는 데에 부족할 수 있다.
-   따라서 굉장히 크게 주는 것 보다는 작게 주는 것이 좋으며, 이를 너무 작게 주어서는 안된다.  
    **논문에 따르면 8보다 크고 32보다 작게 주는 것이 효과적이라고 한다.**

#### Weight Regularization (가중치 규제), Weight Decay (가중치 감소)

-   loss function은 loss 값이 작아지는 방향으로 방향으로 가중치를 update한다.
-   하지만, loss를 줄이는 데에만 신경쓰게 되면, 특정 가중치가 너무 커지면서 오히려 나쁜 결과를 가져올 수 있다.
-   기존 가중치에 특정 연산을 수행하여 loss function의 출력 값과 더해주면 loss function의 결과를 어느정도 제어할 수 있게 된다.
-   보통 파라미터가 많은 Dense Layer에서 많이 사용되고 가중치 규제보다는 loss function에 규제를 걸어 가중치를 감소시키는 원리이다.
-   kerenlregularizer 파라미터에서 l1, l2을 선택할 수 있다.

---

### 실제 영상 데이터를 train, validation, test 데이터 분리

-   아래 code 6~8 참조

#### <div id="cnn-code">CNN Code</div>

<details>
    <summary>1. Funtional API 를 이용한 CNN model 구성.</summary>

        from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten
        from tensorflow.keras.models import Model

        INPUT_SIZE = 28

        # 입력 텐서 정의: 28x28 크기의 gray 이미지
        # 따라서 Input 항목에 3차원으로 입력 이미지의 채널 수를 입력한다
        (단, 3차원으로 나열 되어있을 경우 채널 수 이자 개수를 의미한다.)
        input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))

        ## parms 총 개수
        ## input = 1
        ## kernel = 3 * 3 = 9
        ## filter = 16
        ## 9 * 16 + 16 = 160

        # Conv2D는 2차원 합성곱(Convolution) 레이어를 의미하며 feature map을 생성하기 위한 레이어

        x = Conv2D(filters = 16, kernel_size= 3, strides=1, padding='same',activation='relu')(input_tensor)

        ## input = 16
        ## kernel = 4 * 4 = 16
        ## filter = 32
        ## 16 * 16 * 32 + 32 = 8224

        x = Conv2D(filters = 32, kernel_size= 4, strides=1, padding='same',activation='relu')(x)

        # input = 32
        # kernel = 4 * 4 = 16
        # filter = 64
        # 32 * 16 * 64 + 64 = 32832

        x = Conv2D(filters = 64, kernel_size= 4, strides=1,activation='relu')(x)

        x = MaxPool2D(2)(x)

        # 입력층
        x = Flatten()(x)
        # 히든층
        x = Dense(50, activation='relu')(x)
        # 히든층
        x = Dense(20, activation='relu')(x)
        # 출력층
        output = Dense(10, activation='softmax')(x)

        model = Model(inputs= input_tensor, outputs = output)
        model.summary()

</details>

<details>
    <summary>2. Dropout (뉴런 비활성화 비율).</summary>

        from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Dropout
        from tensorflow.keras.models import Model

        INPUT_SIZE = 28

        input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))


        x = Conv2D(filters = 16, kernel_size= 3, strides=1, padding='same',activation='relu')(input_tensor)
        x = Conv2D(filters = 32, kernel_size= 4, strides=1, padding='same',activation='relu')(x)
        x = Conv2D(filters = 64, kernel_size= 4, strides=1,activation='relu')(x)

        x = MaxPool2D(2)(x)

        x = Flatten()(x)

        # Dropout(rate=비활성화 할 비율 선택)

        x = Dropout(rate=0.5)(x)
        x = Dense(50, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        output = Dense(10, activation='softmax')(x)

        model = Model(inputs= input_tensor, outputs = output)
        model.summary()

</details>

<details>
    <summary>Tip.validation_split</summary>

    # compile 진행 시 별도로 validation 데이터를 구분하지 않고 validation_split을 사용할 수 있다.

    model.fit(x=train_iamges, y=train_target, batch_size=8, epochs=10,
    validation_split=0.2)

</details>

<details>
    <summary>3. RGB 영상 CNN 모델 생성</summary>

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Input, Activation
    from tensorflow.keras.callbacks import Callback

    INPUT_SIZE = 32

    # RGB 영상이기 때문에 최초 3개의 필터를 넣는다.
    input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))

    # padding default == valid
    x = Conv2D(filters = 32, kernel_size=5, padding='valid', activation='relu')(input_tensor)
    x = Conv2D(filters = 32, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(filters = 64, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters = 64, kernel_size=3, padding='same')(x)
    # CNN performance 의 배치 정규화 기능을 사용하기 위해 활성함수를 따로 적용한다.
    x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(filters = 128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters = 128, kernel_size=3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

    x = Flatten(name='classifier_A00_Flatten')(x)
    x = Dropout(name='classifierA_DropOut01', rate=0.5)(x)
    x = Dense(300, activation='relu', name='classifierAD01')(x)
    x = Dropout(name='classifierA_DropOut02', rate=0.5)(x)
    output = Dense(10, activation='softmax', name='output')(x)


    model = Model(inputs = input_tensor, outputs = output)
    model.summary()

</details>

<details>
    <summary>4. keras.losses SparseCategoricalCrossentropy(원-핫 인코딩 후 손실함수 확인)</summary>

    from tensorflow.keras.optimizers import Adam
    # from tensorflow.keras.losses import CategoricalCrossentropy
    # 내가 원-핫 인코딩을 하지않고 함수 내부적으로 원-핫 인코딩을 시켜준다.
    from tensorflow.keras.losses import SparseCategoricalCrossentropy

    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics = ['acc'])

</details>

<details>
    <summary>5. CNN Performance 적용 모델(kernel_initializer(가중치 초기화), BatchNormalization(배치 정규화), GlobalAveragePooling2D), kernel_regularizer(가중치 규제)</summary>

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Activation, Dropout, GlobalAveragePooling2D, BatchNormalization
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.regularizers import l1, l2

    INPUT_SIZE = 32

    input_tensor = Input(shape=(INPUT_SIZE,INPUT_SIZE,3))

    # alpha를 크게 할 수록 Weight값을 작게 만들어서 과적합을 개선할 수 있고,
    # alpha를 작게 할 수록 Weight의 값이 커지지만, 어느 정도 상쇄하므로 과소적합을 개선할 수 있다.
    # 가중치 초기화 (카이밍 히 초기화(he_normal))
    x = Conv2D(filters=64, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(input_tensor)
    # 배치 정규화
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=64, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(300, activation='relu', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
    # 가중치 초기화 (사비에르 글로로트 초기화 (glorot_normal))
    x = Dropout(rate=0.5)(x)
    output = Dense(10, activation='softmax', kernel_initializer='glorot_normal')(x)

    model = Model(inputs= input_tensor, outputs = output)
    model.summary()

</details>

<details>
    <summary>6. ⭐️ (실제 이미지)동물 이미지 영상 train, val, test 구분(MAC, Window)</summary>

    # 사전에 정의된 명칭이 있기 때문에 해당 파일을 불로오는 메소드
    with open('../d_cnn/datasets/animals/translate.py') as f:
        content = f.readline().strip()
        # print(content)
        # 문자열 안에 있는 딕셔너리를 정상적으로 가져오기 위한 메소드 eval
        contents1 = eval(content[content.index("{"):content.index("}") + 1])

        # key, value가 뒤집어져 있는 상태이기 때문에 딕셔너리의 items를 가져와서 key:value 반전
        contents2 = {v: k for k, v in contents1.items()}

    print(contents1, contents2, sep='\n\n')

---

    from glob import glob
    import os

    root = '../d_cnn/datasets/animals/original/'

    # glob 함수의 경우 파일명을 리스트 형식으로 반환하는 함수.
    # os의 root 경로에 있는 모든 파일명을 리스트 형식으로 반환하여 리스트 연결.
    directories = glob(os.path.join(root, '*'))
    print(directories)

    for directory in directories:
        # 플랫폼 독립적으로 디렉토리 이름 추출
        # basname (리눅스에서 파일명이나 확장자를 추출하기 위한 명령어)
        old_name = os.path.basename(directory)

        # 해당 예외처리는 translate.py 항목에 key, value가 중복으로 되어 있어 작성
        try:
            new_name = contents1[old_name]
        except KeyError:
            new_name = contents2.get(old_name, old_name)  # old_name이 contents2에 없으면 그대로 유지
        new_directory = os.path.join(root, new_name)
        os.rename(directory, new_directory)

    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    <!-- window 입니다 -->
    from glob import glob
    import os

    root = './datasets/animals/original/'

    directories = glob(os.path.join(root, '*'))

    for directory in directories:
    # 윈도우의 경우 파일명 앞이 ₩ 로 존재하기 때문에 \\ 를 통하여 해당 파일명을 가져와야 합니다.
        try:
            os.rename(directory, os.path.join(root, contents1[directory[directory.rindex('\\') + 1:]]))
        except KeyError as e:
            os.rename(directory, os.path.join(root, contents2[directory[directory.rindex('\\') + 1:]]))

---

    root = '../d_cnn/datasets/animals/original/'

    # 해당 경로에 있는 전체 경로를 변수에 저장
    directories = glob(os.path.join(root, '*'))
    directory_names = []

    # 반복을 이용한 해당 디렉토리의 파일명을 추출하여 리스트에 저장
    for directory in directories:
        directory_names.append(os.path.basename(directory))

    print(directory_names)

---

    root = '../d_cnn/datasets/animals/original/'

    for name in directory_names:
        for i, file_name in enumerate(os.listdir(os.path.join(root, name))):
            old_file = os.path.join(root + name + '/', file_name)
            new_file = os.path.join(root + name + '/', name + str(i + 1) + '.png')

            # 기존에 있던 파일명을 해당 root 디렉토리의 이름으로 변경 후 뒤에 반복 시 증가되는 숫자 입력
            os.rename(old_file, new_file)

---

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # ImageDataGenerator 인스턴스를 생성
    # 모든 이미지의 픽셀값을 1/255 로 나누어 0과 1사이의 값으로 변환 후 image_data_generator 생성
    image_data_generator = ImageDataGenerator(rescale=1./255)

    # flow_from_directory 메소드를 사용하여 디렉토리에서 이미지 데이터를 로드하고 전처리합니다.
    # flow_from_directory는 ImageDataGenerator의 메소드로 디렉토리 구조에서 이미지 데이터를 로드하고
    # 실시간으로 증강 및 전처리 하는데 사용
    generator = image_data_generator.flow_from_directory(
        root,                 # 이미지가 저장된 디렉토리의 경로
        target_size=(150, 150),  # 모든 이미지를 (150, 150) 크기로 조정합니다.
        batch_size=32,           # 배치 크기를 32로 설정합니다.
        class_mode='categorical' # 클래스 모드를 'categorical'로 설정하여 다중 클래스 분류를 수행합니다.
    )

    # 생성된 클래스 인덱스를 출력합니다.
    # 디렉토리 구조에서 발견된 클래스의 이름과 인덱스를 매핑한 딕셔너리 반환
    print(generator.class_indices)

---

    import pandas as pd
    from sklearn.model_selection import train_test_split

    # flow_from_directory를 이용한 메소드로 해당 파일을 직접 로드하여 경로와 category를 불러옵니다.
    a_df = pd.DataFrame({'file_paths': generator.filepaths, 'targets': generator.classes})
    a_df

---

    # train, validation, test 데이터 분리
    X_train, X_test, y_train, y_test =\
    train_test_split(a_df.file_paths, a_df.targets, stratify=a_df.targets, test_size=0.2, random_state=124)

    print(y_train.value_counts())
    print(y_test.value_counts())

    X_train, X_val, y_train, y_val = \
    train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=124)

    print(y_train.value_counts())
    print(y_val.value_counts())

---

    # 기존 1개의 폴더에 있는 이미지들을 train, validation, test 영상으로 디렉토리 나눠서 copy
    import shutil

    root = '../d_cnn/datasets/animals/'

    for file_path in X_train:
        # animal_dir을 경로 구분자로 분할하여 추출
        # 파일 경로에서 directory 를 추출하려면 dirname 명령어를 사용
        animal_dir = file_path[len(os.path.join(root, 'original')) + 1:file_path.rindex('/')]
        destination = os.path.join(root, 'train', animal_dir)

        # destination 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(destination):
            os.makedirs(destination)

        # 파일을 destination 디렉토리로 복사
        shutil.copy2(file_path, destination)

---

    import shutil

    root = '../d_cnn/datasets/animals/'

    for file_path in X_val:
        # animal_dir을 경로 구분자로 분할하여 추출
        animal_dir = file_path[len(os.path.join(root, 'original')) + 1:file_path.rindex('/')]
        destination = os.path.join(root, 'validation', animal_dir)

        # destination 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(destination):
            os.makedirs(destination)

        # 파일을 destination 디렉토리로 복사
        shutil.copy2(file_path, destination)

---

    root = '../d_cnn/datasets/animals/'

    for file_path in X_test:
        # animal_dir을 경로 구분자로 분할하여 추출
        animal_dir = file_path[len(os.path.join(root, 'original')) + 1:file_path.rindex('/')]
        destination = os.path.join(root, 'test', animal_dir)

        # destination 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(destination):
            os.makedirs(destination)

        # 파일을 destination 디렉토리로 복사
        shutil.copy2(file_path, destination)

</details>
</div>
