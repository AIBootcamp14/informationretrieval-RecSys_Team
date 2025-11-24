# 어떻게 학습데이터를 활용할 수 있을까?

## 개요

대회에서 제공되는 학습데이터는 **질문이 포함되지 않은 순수 과학 상식
문서 약 4200개**로 구성되어 있습니다.\
Retrieval 모델을 학습하기 위해서는 보통 *pair 데이터(Question--Answer,
Document--Question 등)*가 필요하지만, 제공된 데이터에는 이러한 pair
형태가 존재하지 않습니다.

따라서 **직접 pair 데이터를 생성하는 방식**을 활용해야 합니다.\
이 글에서는 대표적인 두 가지 방법을 소개합니다:

-   **Inverse Cloze Task (ICT) 기반 데이터 생성**
-   **Question Generation(QG) 기반 데이터 생성**

------------------------------------------------------------------------

## 1) Inverse Cloze Task (ICT)

Inverse Cloze Task는 문맥에서 특정 문장 또는 단어를 제거하고, AI가 이를
예측하도록 훈련시키는 방식에서 출발합니다.\
Retrieval Task에서는 반대로 **문장을 pseudo-query로 사용**하고,\
해당 문장이 포함된 나머지 문장을 **pseudo evidence**로 활용하여
데이터셋을 구성합니다.

### 예시

원문 Document:

    Zebras have four gaits: walk, trot, canter and gallop. 
    They are generally slower than horses, but their great stamina helps them outrun predators. 
    When chased, a zebra will zigzag from side to side…

ICT를 적용하면 다음과 같은 pair가 생성됩니다:

-   **pseudo-query:** 위 문단에서 임의의 문장\
-   **evidence:** 나머지 문장들\
-   **negative samples:** 다른 Document에서 가져온 무관한 문장들

이렇게 하면 **Retrieval 모델 학습용 쿼리--근거 데이터셋**을 자동으로
만들 수 있습니다.

출처: https://arxiv.org/pdf/1906.00300.pdf

------------------------------------------------------------------------

## 2) Question Generation (QG)

Retrieval 평가에서는 실제 질문 형태의 Query가 필요합니다.\
하지만 ICT 방식은 자연스러운 "질문"을 만들어주지 못합니다.

따라서 **문서를 기반으로 질문을 생성하는 QG 방식**을 사용할 수 있습니다.

### 예시 (PORORO 라이브러리 활용)

``` python
>>> qg = Pororo(task="qg", lang="ko")
>>> qg("카카오톡", 
      "카카오톡은 스마트폰의 데이터 통신 기능을 이용하여 문자 과금 없이 메시지를 주고받을 수 있는 앱이다.",
      n_wrong=3)
```

생성 결과:

    ('스마트폰의 데이터 통신 기능을 이용해서 문자 과금 없이 사람들과 메시지를 주고받을 수 있는 앱을 뭐라고 해?', 
     ['텔레그램', '챗온', '디스코드'])

QG에서 고려할 점: - 어떤 문장을 "정답(answer span)"으로 지정할 것인가? -
생성된 질문이 자연스러운가? - 문맥과 적절히 연결되는가?

이러한 점 때문에 **생성 품질 평가도 병행해야** 합니다.

------------------------------------------------------------------------

## 결론

학습데이터에는 Query--Document--Answer 구조가 없기 때문에\
**Inverse Cloze Task**와 **Question Generation**을 통해 pair 데이터를
직접 만들어야 합니다.

-   ICT: 자동으로 pseudo-query를 뽑아 Retrieval용 데이터 구성\
-   QG: 자연스러운 질문 생성 → 대회 평가 포맷과 더 잘 맞음

두 방법을 조합하면 **더 강력한 Retrieval 학습 데이터셋**을 구축할 수
있습니다.

------------------------------------------------------------------------

더 발전된 아이디어나 다른 데이터 생성 전략이 있다면 함께 나누어 보는
것도 좋겠습니다!
