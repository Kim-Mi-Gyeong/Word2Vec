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
   
   

   
