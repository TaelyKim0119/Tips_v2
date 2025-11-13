"""
한국어 감정 분석 모듈
- KnuSentiLex 사전 기반 감정 분석
- 하이브리드 모델 (KR-FinBERT + KnuSentiLex)
- ALS 모델 지원
"""

import os
import subprocess
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from kiwipiepy import Kiwi
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:
    """한국어 감정 분석기"""
    
    def __init__(
        self,
        base_model_candidates: Optional[List[str]] = None,
        als_model: str = "alsgyu/sentiment-analysis-fine-tuned-model",
        lexicon_path: Optional[str] = None
    ):
        """
        Args:
            base_model_candidates: 기본 모델 후보 리스트
            als_model: ALS 모델 이름
            lexicon_path: KnuSentiLex 사전 파일 경로
        """
        # 기본 설정
        self.base_model_candidates = base_model_candidates or [
            "snunlp/KR-FinBERT",
            "nlpai-lab/kcelectra-base-kor-finetuned-nsmc",
            "beomi/KcELECTRA-base"
        ]
        self.als_model = als_model
        
        # KnuSentiLex 사전 로드
        self.lexicon_path = lexicon_path or self._ensure_lexicon_file()
        self.lexicon = self._load_knu_lexicon(self.lexicon_path)
        print(f"[LOAD] KnuSentiLex entries: {len(self.lexicon):,}")
        
        # 형태소 분석기 초기화
        self.kiwi = Kiwi()
        
        # 모델 로드
        self.base_name, self.base_tok, self.base_mdl, self.base_labels = self._try_load_any(self.base_model_candidates)
        self.als_tok, self.als_mdl, self.als_labels = self._load_classifier(self.als_model)
        print(f"[LOAD] Base model: {self.base_name} -> {self.base_labels}")
        print(f"[LOAD] ALS model: {self.als_model} -> {self.als_labels}")
    
    def _ensure_lexicon_file(self) -> str:
        """KnuSentiLex 사전 파일 확보"""
        raw_urls = [
            "https://raw.githubusercontent.com/park1200656/knu_senti_dict/master/SentiWord_Dict.txt",
            "https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/SentiWord_Dict.txt",
        ]
        local_path = "SentiWord_Dict.txt"
        
        # 1) raw 파일 다운로드 시도
        for url in raw_urls:
            try:
                rc = subprocess.call(
                    ["curl", "-L", "-o", local_path, url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if rc == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    print(f"[KNU] Downloaded from: {url}")
                    return local_path
            except Exception:
                pass
        
        # 2) 로컬 파일 확인
        candidates = [
            "SentiWord_Dict.txt",
            "knusentilexdownload/SentiWord_Dict.txt",
            "knusentilexdownload/KnuSentiLex/SentiWord_Dict.txt",
        ]
        for path in candidates:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                print(f"[KNU] Found local file: {path}")
                return path
        
        raise FileNotFoundError("KnuSentiLex 사전 파일을 찾을 수 없습니다.")
    
    def _load_knu_lexicon(self, path: str) -> Dict[str, float]:
        """KnuSentiLex 사전 로드"""
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["word", "polarity"],
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip"
        )
        df["polarity"] = pd.to_numeric(df["polarity"], errors="coerce")
        df = df.dropna(subset=["word", "polarity"])
        df["word"] = df["word"].astype(str).str.strip()
        return dict(zip(df["word"], df["polarity"]))
    
    def _load_classifier(self, model_name: str) -> Tuple:
        """분류 모델 로드"""
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        with torch.no_grad():
            dummy = tok("테스트", return_tensors="pt", truncation=True, padding=True)
            n_cls = mdl(**dummy).logits.shape[-1]
        if n_cls == 2:
            labels = ["부정", "긍정"]
        elif n_cls == 3:
            labels = ["부정", "중립", "긍정"]
        else:
            labels = [f"class_{i}" for i in range(n_cls)]
        return tok, mdl, labels
    
    def _try_load_any(self, candidates: List[str]) -> Tuple:
        """후보 모델 중 하나 로드"""
        last_error = None
        for name in candidates:
            try:
                tok, mdl, labels = self._load_classifier(name)
                return name, tok, mdl, labels
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"모델을 로드할 수 없습니다. 마지막 오류: {last_error}")
    
    def lexicon_score(self, text: str) -> float:
        """텍스트의 lexicon 기반 감정 점수 계산"""
        tokens = [t.form for t in self.kiwi.tokenize(text)]
        score = 0.0
        for t in tokens:
            key = t.strip()
            if key in self.lexicon:
                score += float(self.lexicon[key])
        # 원문 그대로 매칭 (이모티콘 등)
        for raw_piece in text.split():
            key = raw_piece.strip()
            score += float(self.lexicon.get(key, 0.0))
        return score
    
    @torch.no_grad()
    def _predict_logits(self, tok, mdl, labels, text: str) -> Tuple[str, float, Dict[str, float]]:
        """모델 예측"""
        x = tok(text, return_tensors="pt", truncation=True, padding=True)
        logits = mdl(**x).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
        idx = int(torch.argmax(logits, dim=-1).item())
        return (
            labels[idx],
            float(probs[idx]),
            {labels[i]: float(p) for i, p in enumerate(probs)}
        )
    
    def _sarcasm_flag(self, text: str, lscore: float) -> bool:
        """반어/비꼼 감지"""
        pos_markers = {"천재", "대단", "역시", "완벽", "잘했다", "축하"}
        has_pos = any(w in text for w in pos_markers)
        punct = (text.count("...") >= 2) or ("?!" in text)
        return (has_pos and lscore < -0.3) or punct
    
    def _signed_from_dist(self, dist: Dict[str, float]) -> float:
        """분포에서 부호 있는 점수 계산"""
        p_pos = dist.get("긍정", 0.0)
        p_neg = dist.get("부정", 0.0)
        p_neu = dist.get("중립", 0.0)
        signed = p_pos - p_neg
        if p_neu > max(p_pos, p_neg):
            signed *= (1.0 - p_neu)
        return signed
    
    def hybrid_predict(
        self,
        text: str,
        w_model: float = 0.7,
        w_lex: float = 0.3
    ) -> Dict:
        """
        하이브리드 감정 예측 (KR-FinBERT + KnuSentiLex)
        
        Args:
            text: 분석할 텍스트
            w_model: 모델 가중치
            w_lex: lexicon 가중치
        
        Returns:
            감정 분석 결과 딕셔너리
        """
        lscore = self.lexicon_score(text)
        base_label, base_conf, base_dist = self._predict_logits(
            self.base_tok, self.base_mdl, self.base_labels, text
        )
        signed = self._signed_from_dist(base_dist)
        final = w_model * signed + w_lex * lscore
        
        if "삭제된 댓글" in text or "부적절한 표현" in text:
            label = "기타"
        elif self._sarcasm_flag(text, lscore):
            label = "반어/비꼼"
        else:
            if final > 0.2:
                label = "긍정"
            elif final < -0.2:
                label = "분노(강부정)" if (lscore < -1.0 or base_dist.get("부정", 0) > 0.85) else "부정"
            else:
                label = "중립"
        
        return {
            "label": label,
            "score": round(final, 3),
            "lex_score": round(lscore, 3),
            "base_label": base_label,
            "base_conf": round(base_conf, 3),
            "base_dist": {k: round(v, 3) for k, v in base_dist.items()}
        }
    
    def als_predict(self, text: str) -> Dict:
        """
        ALS 모델 기반 감정 예측
        
        Args:
            text: 분석할 텍스트
        
        Returns:
            감정 분석 결과 딕셔너리
        """
        label, conf, dist = self._predict_logits(
            self.als_tok, self.als_mdl, self.als_labels, text
        )
        return {
            "label": label,
            "confidence": round(conf, 3),
            "scores": {k: round(v, 3) for k, v in dist.items()}
        }
    
    def predict(self, text: str, method: str = "hybrid") -> Dict:
        """
        감정 예측 (통합 인터페이스)
        
        Args:
            text: 분석할 텍스트
            method: "hybrid" 또는 "als"
        
        Returns:
            감정 분석 결과
        """
        if method == "hybrid":
            return self.hybrid_predict(text)
        elif method == "als":
            return self.als_predict(text)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'hybrid' or 'als'")
    
    def batch_predict(self, texts: List[str], method: str = "hybrid") -> List[Dict]:
        """배치 예측"""
        return [self.predict(text, method) for text in texts]


def predict_sentiment_simple(text: str, model_name: str = "alsgyu/sentiment-analysis-fine-tuned-model") -> Dict:
    """
    간단한 감정 분석 함수 (ALS 모델만 사용)
    
    Args:
        text: 분석할 텍스트
        model_name: 모델 이름
    
    Returns:
        감정 분석 결과
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    label_id = torch.argmax(probs).item()
    
    labels = ["부정", "중립", "긍정"] if probs.shape[0] == 3 else ["부정", "긍정"]
    
    return {
        "text": text,
        "label": labels[label_id],
        "confidence": round(float(probs[label_id].detach()), 3),
        "scores": {labels[i]: round(float(p.detach()), 3) for i, p in enumerate(probs)}
    }
