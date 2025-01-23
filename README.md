# 머신러닝, 딥러닝


2025.01.21
1. Overfitting(과적합)을 막는 방법 : 데이터 양 늘리기, 모델 복잡도 줄이기, Regularization(가중치 규제) 적용, Dropout
2. NLP(자연어)
   2.1. One-hot encoding
   2.2. Word Embedding : 희소표현, 밀집표현, Word Embedding, 원-핫 벡터, 임베딩 벡터
   2.3. Word2Vec : 희소표현, 분산표현, CBOW, Skip-gram, Negiative Sampling
   2.4. Word2Vec 학습 및 시각화
     2.4.1. 네이버 영화 리뷰 데이터로 한국어 Word2Vec을 만들어 보기

2025.01.22
1. RNN
   1.1. 단어에 대한 인덱스 부여 : 빈도수 순대로 정렬 후 순차적으로 인덱스 부여 --> 빈도수 적은 단어 제거
   1.2. RNN : 이진분류-시그모이드함수, 다중 클래스 분류-소프트맥스 함수
2. LSTM
   2.1. 
3. Youtube 크롤링
   3.1. 구글 API
   3.2. 크롤링 : video_id 100개, 한국댓글 100개 --> 쿼리 날림
                (error)
                  1) !python -m gensim.scripts.word2vec2tensor --input ko_w2v --output ko_w2v --> ValueError: could not broadcast input array from shape (0,) into shape                            (100,)
                     파이썬에서 배열(array)을 처리할 때 발생하는 오류로, 배열의 크기(shape)가 맞지 않아 데이터를 재배치(broadcast)할 수 없을 때 발생.
                     이 경우, 입력 배열의 크기가 (0,)이므로 빈 배열(empty array)을 처리하려다 오류가 발생.

  <소스 수정> 
   sentence = str(sentence).strip() --> 양쪽 공백 삭제
   if not word in stopwords # 조건1 --> 조건추가
   and len(word) >= 2 # 조건2 --> 조건추가
   and word.isalpha() ] # 한글이나 영어만 --> 조건추가

  <전체 소스>
   stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

   tokenized_data = []
   # merged_df['text'] => comments['Comment']
   for sentence in tqdm.tqdm(comments['Comment']):
       sentence = str(sentence).strip()

       if not sentence:
        continue
   
       tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
       stopwords_removed_sentence = [word for word in tokenized_sentence
                                     if not word in stopwords # 조건1 
                                        and len(word) >= 2 # 조건2
                                        and word.isalpha() ] # 한글이나 영어만 
                                     
       if stopwords_removed_sentence: # 빈리스트가 아니라면 추가
           tokenized_data.append(stopwords_removed_sentence)

      ==> 단일쿼리시 크롤링 가능

   4. 주식예측

2025.01.23
(오전)
1. Dacon
   - 교차검증 cross validation(cv) : Training set과 Validation 을 여러번 나눈 뒤 모델의 학습을 검증하는 방식
   - stratifiy : train_test_split 균일하게 데이터 배분
  
2. 참고 소스
   - pip install -q konlpy : 조용히해!, 잡다한거 나올때 안나오게 하기
  

3. LSTM을 이용한 네이버 영화리뷰
   - https://wikidocs.net/217687 참고
   - labeling : 수작업함
   - 토근화 : Okt

4. GPT-2를 이용한 한국어 뉴스 긍정, 부정 감성 분류 --> Transformer 중의 일부
   - https://wikidocs.net/217619 참고
   
GPT를 사용하기 위해서는 토크나이저와 모델이 반드시 맵핑 관계여야만 합니다. 다시 말해 아래의 이름에 들어가는 모델이름은 반드시 동일해야 합니다.

- AutoTokenizer.from_pretrained('모델이름')
- AutoModelForSequenceClassification.from_pretrained("모델이름")

토크나이저는 내부적으로 Vocabulary를 갖고 있어 정수 인코딩을 수행해주는 모듈입니다.

# 한국어 GPT 중 하나인 'skt/kogpt2-base-v2'를 사용.
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')

토크나이저를 로드하고 .tokenize()를 이용하여 텍스트를 토큰화하고 토큰화 결과를 convert_tokens_to_ids()를 사용하여 정수 인코딩 할 수 있습니다. 이러한 정수 인코딩 과정을 data_to_tensor()라는 이름의 함수로 구현합니다. data_to_tensor() 함수는 정수 인코딩 뿐만 아니라 최대 길이를 입력받으면, 최대 길이까지 패딩(padding)을 수행하도록 하고, 어텐션 마스크(attention mask)도 계산하여 같이 반환하도록 합니다.

   - 추론하기(inference) : 임의의 데이터에 대해서 예측 결과를 얻으려면 예측 함수를 만들어야 합니다. transformers 패키지에서는 이러한 과정을 자동으로 해주는 pipeline 도구를 제공합니다.                            현재 풀고자 하는 문제가 어떤 문제인지, 모델은 무엇인지, 토크나이저는 무엇인지를 알려주면 임의의 입력에 대해서 예측을 할 수 있게됩니다.

(오후)
5. Faiss와 SBERT를 이용한 시맨틱 검색기(Semantic Search) => 추천리스트
   - https://wikidocs.net/217139
   - SBERT?
   - distilbert-base-nli-mean-tokens : Hugging Face의 Sentence Transformers 라이브러리에서 제공하는 사전 학습된 모델 중 하나입니다. 주로 문장 임베딩(sentence embedding)을 생성하                                         는데 사용되며, 특히 자연어 추론(NLI) 데이터셋으로 학습되었습니다.
   - 코사인 유사도를 이용한 추천 시스템
     -> https://wikidocs.net/217505

==> 데이터의 순서가 있는 것을 처리하는 모델 : RNN, LSTM, GPT(Transformer) => [긍정, 부정]


6. 챗봇실습
   - HuggingFace API key 받기: https://sunshower99.tistory.com/30 참고
   - 
   - Use Gemini+RAG >>>> Gemini 키 발급: https://languagestory.tistory.com/315
  

   

   
