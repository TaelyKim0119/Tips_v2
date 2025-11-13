"""
감정 분석 사용 예제 스크립트
"""

from sentiment_analyzer import SentimentAnalyzer, predict_sentiment_simple
from data_utils import load_comments, add_sentiment_labels, get_sentiment_summary, save_results
import pandas as pd


def example_simple_analysis():
    """간단한 감정 분석 예제"""
    print("=" * 60)
    print("예제 1: 간단한 감정 분석")
    print("=" * 60)
    
    texts = [
        "이 서비스 정말 좋아요!",
        "이건 좀 아쉽네요",
        "중간 정도인 것 같아요",
        "와, 진짜 천재시네요. 이렇게 완벽한 실수는 처음 봅니다.",
        "작성자에 의해 삭제된 댓글입니다."
    ]
    
    for text in texts:
        result = predict_sentiment_simple(text)
        print(f"\n텍스트: {text}")
        print(f"감정: {result['label']}, 신뢰도: {result['confidence']}")
        print(f"세부 점수: {result['scores']}")


def example_hybrid_analysis():
    """하이브리드 모델 예제"""
    print("\n" + "=" * 60)
    print("예제 2: 하이브리드 모델 분석")
    print("=" * 60)
    
    analyzer = SentimentAnalyzer()
    
    texts = [
        "KT 해킹 사태 정말 심각하네요",
        "서비스 개선 잘 되고 있어요",
        "돈만 받아먹을 욕지 보상해줄 생각도 없는 개그지 같은 놈들"
    ]
    
    for text in texts:
        result = analyzer.hybrid_predict(text)
        print(f"\n텍스트: {text}")
        print(f"감정: {result['label']}")
        print(f"종합 점수: {result['score']}")
        print(f"Lexicon 점수: {result['lex_score']}")
        print(f"Base 모델: {result['base_label']} (신뢰도: {result['base_conf']})")


def example_batch_processing():
    """배치 처리 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 배치 처리")
    print("=" * 60)
    
    # 샘플 데이터 생성
    sample_data = {
        "video_id": ["v1", "v1", "v2", "v2", "v3"],
        "text": [
            "역시 대단하다 또 유출이네 ㅋㅋ",
            "서비스 좋아요. 개선도 빨라졌고요",
            "아 이건 좀 아닌 듯",
            "업데이트 나쁘지 않네요",
            "최악이다 진심 실망"
        ],
        "likeCount": [10, 5, 3, 8, 2]
    }
    
    df = pd.DataFrame(sample_data)
    print("\n원본 데이터:")
    print(df)
    
    # 감정 분석 추가
    analyzer = SentimentAnalyzer()
    df_labeled = add_sentiment_labels(df, text_column="text", analyzer=analyzer, method="hybrid")
    
    print("\n감정 분석 결과:")
    print(df_labeled[["text", "text_label", "fused_score"]])
    
    # 요약 통계
    print("\n감정별 통계:")
    summary = get_sentiment_summary(df_labeled)
    print(summary)


def example_file_processing():
    """파일 처리 예제 (실제 파일이 있을 경우)"""
    print("\n" + "=" * 60)
    print("예제 4: 파일 처리 (주석 처리됨)")
    print("=" * 60)
    
    # 실제 파일 경로로 변경하여 사용
    """
    file_path = "YouTube_결과/GS리테일_정보유출_comments_20251103_154830.xlsx"
    
    try:
        # 데이터 로드
        df = load_comments(file_path, text_column="text")
        print(f"로드된 데이터: {len(df)}건")
        
        # 감정 분석
        analyzer = SentimentAnalyzer()
        df_labeled = add_sentiment_labels(df, text_column="text", analyzer=analyzer)
        
        # 결과 저장
        output_path = "output_with_sentiment.xlsx"
        save_results(df_labeled, output_path)
        
        # 요약
        summary = get_sentiment_summary(df_labeled)
        print("\n감정별 통계:")
        print(summary)
        
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        print(f"오류 발생: {e}")
    """
    print("실제 파일 경로를 지정하여 사용하세요.")


def example_model_comparison():
    """모델 비교 예제"""
    print("\n" + "=" * 60)
    print("예제 5: 모델 비교")
    print("=" * 60)
    
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "이 서비스 정말 좋아요!",
        "이건 좀 아쉽네요",
        "중간 정도인 것 같아요"
    ]
    
    for text in test_texts:
        print(f"\n텍스트: {text}")
        
        # 하이브리드 모델
        hybrid_result = analyzer.hybrid_predict(text)
        print(f"  하이브리드: {hybrid_result['label']} (점수: {hybrid_result['score']})")
        
        # ALS 모델
        als_result = analyzer.als_predict(text)
        print(f"  ALS 모델: {als_result['label']} (신뢰도: {als_result['confidence']})")


if __name__ == "__main__":
    print("감정 분석 예제 실행\n")
    
    # 예제 1: 간단한 분석
    example_simple_analysis()
    
    # 예제 2: 하이브리드 모델
    example_hybrid_analysis()
    
    # 예제 3: 배치 처리
    example_batch_processing()
    
    # 예제 4: 파일 처리 (주석 처리됨)
    example_file_processing()
    
    # 예제 5: 모델 비교
    example_model_comparison()
    
    print("\n" + "=" * 60)
    print("모든 예제 실행 완료!")
    print("=" * 60)
