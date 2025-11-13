# 한국어 감정 분석 프로젝트

YouTube 댓글, 네이버 뉴스/댓글 등 다양한 소스의 한국어 텍스트에 대한 감정 분석 프로젝트입니다.

## 주요 기능

- **하이브리드 감정 분석**: KR-FinBERT + KnuSentiLex 사전 기반 감정 분석
- **ALS 모델 지원**: alsgyu/sentiment-analysis-fine-tuned-model 사용
- **배치 처리**: 대량의 댓글 데이터 일괄 분석
- **다양한 데이터 소스**: YouTube, 네이버, 비즈니스 데이터 지원

## 프로젝트 구조

```
.
├── sentiment_analyzer.py    # 감정 분석 모듈
├── data_utils.py            # 데이터 처리 유틸리티
├── PoC_v2_youtube.ipynb     # YouTube 데이터 수집 및 분석
├── PoC_v2_naver.ipynb       # 네이버 데이터 분석
├── PoC_v2_biz.ipynb         # 비즈니스 데이터 분석
├── PoC_v2_댓글감정.ipynb    # 감정 분석 실험
├── SentiWord_Dict.txt       # KnuSentiLex 사전
├── YouTube_결과/            # YouTube 분석 결과
├── 공유폴더(딥테크)/         # 공유 데이터
└── 데이터/                  # 회사별 데이터 (GS, KT, SKT, 롯데카드)
```

## 설치 방법

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. KnuSentiLex 사전 다운로드

사전 파일은 자동으로 다운로드됩니다. 수동으로 다운로드하려면:

```bash
git clone https://github.com/park1200656/KnuSentiLex.git knusentilexdownload
```

또는

```bash
curl -L -o SentiWord_Dict.txt https://raw.githubusercontent.com/park1200656/knu_senti_dict/master/SentiWord_Dict.txt
```

## 사용 방법

### 기본 사용법

```python
from sentiment_analyzer import SentimentAnalyzer

# 분석기 초기화
analyzer = SentimentAnalyzer()

# 단일 텍스트 분석
result = analyzer.predict("이 서비스 정말 좋아요!")
print(result)
# {'label': '긍정', 'score': 0.523, ...}

# 하이브리드 모델 사용
result = analyzer.hybrid_predict("이건 좀 아쉽네요")
print(result)
# {'label': '부정', 'score': -0.234, ...}

# ALS 모델 사용
result = analyzer.als_predict("중간 정도인 것 같아요")
print(result)
# {'label': '중립', 'confidence': 0.678, ...}
```

### 배치 처리

```python
from data_utils import load_comments, add_sentiment_labels, save_results

# 데이터 로드
df = load_comments("YouTube_결과/GS리테일_정보유출_comments_20251103_154830.xlsx")

# 감정 분석 추가
df_labeled = add_sentiment_labels(df, text_column="text", method="hybrid")

# 결과 저장
save_results(df_labeled, "output_with_sentiment.xlsx")
```

### 간단한 감정 분석 (ALS 모델만)

```python
from sentiment_analyzer import predict_sentiment_simple

result = predict_sentiment_simple("정말 만족스러워요!")
print(f"감정: {result['label']}, 신뢰도: {result['confidence']}")
```

## 감정 라벨

- **긍정**: 긍정적인 감정
- **부정**: 부정적인 감정
- **중립**: 중립적인 감정
- **분노(강부정)**: 매우 강한 부정 감정
- **반어/비꼼**: 반어법이나 비꼼 표현
- **기타**: 삭제된 댓글 등 특수 케이스

## 모델 정보

### 하이브리드 모델
- **Base Model**: KR-FinBERT (snunlp/KR-FinBERT)
- **Lexicon**: KnuSentiLex 사전
- **가중치**: 모델 70% + 사전 30%

### ALS 모델
- **Model**: alsgyu/sentiment-analysis-fine-tuned-model
- **출력**: 부정, 중립, 긍정 (3-class)

## 데이터 소스

### YouTube
- GS리테일 정보유출
- KT 펨토셀 해킹
- SKT 유심칩 유출
- 롯데카드 정보유출
- 카카오 개인정보 유출

### 네이버
- 뉴스 기사
- 댓글 데이터

### 비즈니스
- 블로그 데이터
- 트위터 데이터

## 예제 노트북

- `PoC_v2_youtube.ipynb`: YouTube 댓글 수집 및 감정 분석
- `PoC_v2_naver.ipynb`: 네이버 데이터 분석
- `PoC_v2_댓글감정.ipynb`: 감정 분석 모델 비교 실험

## 주의사항

1. **모델 다운로드**: 첫 실행 시 Hugging Face에서 모델을 다운로드하므로 시간이 걸릴 수 있습니다.
2. **메모리**: 대용량 데이터 처리 시 충분한 메모리가 필요합니다.
3. **GPU**: GPU가 있으면 더 빠른 처리가 가능합니다 (PyTorch 자동 감지).

## 라이선스

이 프로젝트에서 사용하는 모델 및 사전:
- KR-FinBERT: Apache 2.0
- KnuSentiLex: MIT License
- alsgyu/sentiment-analysis-fine-tuned-model: 해당 모델의 라이선스 참조

## 참고 자료

- [KnuSentiLex](https://github.com/park1200656/KnuSentiLex)
- [KR-FinBERT](https://huggingface.co/snunlp/KR-FinBERT)
- [ALS Sentiment Model](https://huggingface.co/alsgyu/sentiment-analysis-fine-tuned-model)
