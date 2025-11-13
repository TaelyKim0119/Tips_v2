"""
데이터 처리 유틸리티 모듈
- CSV/Excel 파일 읽기/쓰기
- 데이터 전처리
- 감정 분석 결과 추가
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from tqdm import tqdm
from sentiment_analyzer import SentimentAnalyzer


def load_comments(file_path: str, text_column: str = "text") -> pd.DataFrame:
    """
    댓글 데이터 로드
    
    Args:
        file_path: 파일 경로 (CSV 또는 Excel)
        text_column: 텍스트 컬럼 이름
    
    Returns:
        DataFrame
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_path}")
    
    if text_column not in df.columns:
        raise ValueError(f"컬럼 '{text_column}'을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
    
    return df


def add_sentiment_labels(
    df: pd.DataFrame,
    text_column: str = "text",
    analyzer: Optional[SentimentAnalyzer] = None,
    method: str = "hybrid",
    show_progress: bool = True
) -> pd.DataFrame:
    """
    DataFrame에 감정 라벨 추가
    
    Args:
        df: 입력 DataFrame
        text_column: 텍스트 컬럼 이름
        analyzer: SentimentAnalyzer 인스턴스 (None이면 새로 생성)
        method: "hybrid" 또는 "als"
        show_progress: 진행 상황 표시 여부
    
    Returns:
        감정 라벨이 추가된 DataFrame
    """
    if analyzer is None:
        analyzer = SentimentAnalyzer()
    
    results = []
    texts = df[text_column].astype(str)
    iterator = tqdm(texts, desc="감정 분석 중") if show_progress else texts
    
    for text in iterator:
        try:
            result = analyzer.predict(text, method=method)
            if method == "hybrid":
                results.append({
                    "text_label": result["label"],
                    "fused_score": result["score"],
                    "lex_score": result.get("lex_score", 0.0),
                    "base_label": result.get("base_label", ""),
                })
            else:  # als
                results.append({
                    "text_label": result["label"],
                    "confidence": result["confidence"],
                    "fused_score": result["scores"].get("긍정", 0.0) - result["scores"].get("부정", 0.0),
                })
        except Exception as e:
            results.append({
                "text_label": "오류",
                "fused_score": 0.0,
                "error": str(e)
            })
    
    res_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), res_df], axis=1)
    return df


def save_results(df: pd.DataFrame, output_path: str):
    """
    결과 저장
    
    Args:
        df: 저장할 DataFrame
        output_path: 출력 파일 경로
    """
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    elif output_path.endswith(('.xlsx', '.xls')):
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {output_path}")
    
    print(f"✅ 결과 저장 완료: {output_path}")


def get_sentiment_summary(df: pd.DataFrame, label_column: str = "text_label") -> pd.DataFrame:
    """
    감정 분석 결과 요약
    
    Args:
        df: DataFrame
        label_column: 감정 라벨 컬럼 이름
    
    Returns:
        감정별 통계 DataFrame
    """
    if label_column not in df.columns:
        raise ValueError(f"컬럼 '{label_column}'을 찾을 수 없습니다.")
    
    summary = df[label_column].value_counts().reset_index()
    summary.columns = ["감정", "개수"]
    summary["비율(%)"] = (summary["개수"] / len(df) * 100).round(2)
    
    return summary


def filter_deleted_comments(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    삭제된 댓글 필터링
    
    Args:
        df: DataFrame
        text_column: 텍스트 컬럼 이름
    
    Returns:
        필터링된 DataFrame
    """
    deleted_keywords = [
        "작성자에 의해 삭제",
        "삭제된 댓글",
        "클린봇",
        "운영규정 미준수",
        "부적절한 표현"
    ]
    
    mask = df[text_column].astype(str).apply(
        lambda x: not any(keyword in x for keyword in deleted_keywords)
    )
    
    return df[mask].reset_index(drop=True)


def clean_text(text: str) -> str:
    """
    텍스트 정리
    
    Args:
        text: 원본 텍스트
    
    Returns:
        정리된 텍스트
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    # HTML 태그 제거
    import re
    text = re.sub(r'<[^>]+>', '', text)
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
