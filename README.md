# Deep Learning

-   인공 신경망(Artivicial Neural Network)의 층을 연속적으로 깊게 쌓아올려 데이터를 학습하는 방식을 의미한다.
-   인간이 학습하고 기억하는 매커니즘을 모방한 기계학습이다.
-   인간은 학습 시, 뇌에 있는 뉴런이 자극을 받아들여서 일정 자극 이상이 되면, 화학물질을 통해 다른 뉴런과 연결되며 해당 부분이 발달한다.
-   자극이 약하거나 기준치를 넘지 못하면, 뉴런은 연결되지 않는다.
-   입력한 데이터가 활성 함수에서 임계점을 넘게 되면 출력된다.
-   초기 인공 신경망(Perceptron)에서 깊게 층을 쌓아 학습하는 딥 러닝으로 발전한다.
-   딥 러닝은 Input nodes layer, Hidden nodes layer, Output nodes layer, 이렇게 세 가지 층이 존재한다.

## 목차

1. <details>
       <summary>perceptron</summary>
       <ul>
           <li>SLP
               <details>
                   <summary>theory & Code</summary>
                   <ul>
                       <li><a href="#slp-theory">이론</a></li>
                       <li><a href="#slp-code">주요 코드</a></li>
                   </ul>
               </details>
           </li>
           <li>MLP
               <details>
                   <summary>theory & Code</summary>
                   <ul>
                       <li><a href="#mlp-theory">이론</a></li>
                       <li><a href="#mlp-code">주요 코드</a></li>
                   </ul>
               </details>
           </li>
           <li>activation_function
               <details>
                   <summary>theory & Code</summary>
                   <ul>
                       <li><a href="#act-theory">이론</a></li>
                       <li><a href="#act-code">주요 코드</a></li>
                   </ul>
               </details>
           </li>
           <li>optimizer
               <details>
                   <summary>theory & Code</summary>
                   <ul>
                       <li><a href="#opti-theory">이론</a></li>
                       <li><a href="#opti-code">주요 코드</a></li>
                   </ul>
               </details>
           </li>
       </ul>
   </details>
2. [tensorflow](#tensorflow)
3. [CNN](#cnn)

---

## perceptron

### SLP

#### <div id="slp-theory">SLP 이론</div>

SLP 관련 이론 내용

#### <div id="slp-code">SLP Code</div>

SLP 관련 코드 내용

### MLP

#### <div id="mlp-theory">MLP 이론</div>

MLP 관련 이론 내용

#### <div id="mlp-code">MLP Code</div>

MLP 관련 코드 내용

### activation_function

#### <div id="act-theory">Activation Function 이론</div>

Activation Function 관련 이론 내용

#### <div id="act-code">Activation Function Code</div>

Activation Function 관련 코드 내용

### optimizer

#### <div id="opti-theory">Optimizer 이론</div>

Optimizer 관련 이론 내용

#### <div id="opti-code">Optimizer Code</div>

Optimizer 관련 코드 내용

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
        # Dropout - 비활성화 할 비율 선택
        x = Dropout(rate=0.5)(x)
        x = Dense(50, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        output = Dense(10, activation='softmax')(x)

        model = Model(inputs= input_tensor, outputs = output)
        model.summary()

</details>

</div>
