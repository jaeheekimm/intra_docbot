import os
import json
import pickle
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from kiwipiepy import Kiwi
from langsmith import traceable

load_dotenv()

from src.utils.paths import CHROMA_DIR, CHROMA_COLLECTION as COLLECTION, BM25_PATH, EMBED_MODEL

# 매 호출마다 생성하면 느려서 전역으로 1회만 생성
_KIWI = Kiwi()


def _import_chroma():
    """langchain_chroma → langchain_community 순으로 fallback"""
    try:
        from langchain_chroma import Chroma  # type: ignore

        return Chroma
    except Exception:
        from langchain_community.vectorstores import Chroma  # type: ignore

        return Chroma


def tokenize(text: str) -> List[str]:
    """Kiwi 형태소 분석으로 명사(N)/동사(V)만 추출. bm25_index.py와 동일한 로직 유지"""
    tokens = []
    for tok in _KIWI.tokenize(text):
        if tok.tag.startswith("N") or tok.tag.startswith("V"):
            tokens.append(tok.form)
    return tokens


def minmax_norm(scores: List[float]) -> List[float]:
    """점수 리스트를 min-max 방식으로 0~1 정규화. 점수 차이를 그대로 반영"""
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-9:
        return [1.0 if s == max_s else 0.0 for s in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


def rank_norm(scores: List[float], reverse: bool = True) -> List[float]:
    """순위 기반 정규화. 1등=1.0, 꼴찌=0.0

    - reverse=True: 점수 클수록 좋음 (유사도)
    - reverse=False: 점수 작을수록 좋음 (거리)
    """
    if not scores:
        return scores

    indexed = list(enumerate(scores))

    if reverse:
        sorted_idx = sorted(indexed, key=lambda x: x[1], reverse=True)
    else:
        sorted_idx = sorted(indexed, key=lambda x: x[1])

    n = len(scores)
    out = [0.0] * n

    for rank, (orig_i, _) in enumerate(sorted_idx):
        if n == 1:
            out[orig_i] = 1.0
        else:
            out[orig_i] = 1.0 - (rank / (n - 1))

    return out


def safe_json_loads(x: Any) -> Any:
    """문자열이면 JSON 파싱 시도. 실패하거나 문자열이 아니면 그대로 반환"""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


class HybridRetriever:
    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        collection: str = COLLECTION,
        bm25_path: str = BM25_PATH,
        embed_model: str = EMBED_MODEL,
    ):
        Chroma = _import_chroma()
        self.embeddings = OpenAIEmbeddings(model=embed_model)

        self.db = Chroma(
            collection_name=collection,
            embedding_function=self.embeddings,
            persist_directory=chroma_dir,
        )

        if not os.path.exists(bm25_path):
            raise RuntimeError(
                f"{bm25_path} 없음. python src/pipeline/bm25_index.py 먼저 실행해 주세요."
            )

        with open(bm25_path, "rb") as f:
            payload = pickle.load(f)

        self.bm25 = payload["bm25"]
        self.bm25_chunk_ids = payload["chunk_ids"]
        self.bm25_texts = payload["texts"]
        self.bm25_metas = payload["metadatas"]

    @traceable(name="Retriever.DenseSearch")
    def dense_search(
        self, query: str, k: int
    ) -> List[Tuple[str, str, Dict[str, Any], float]]:
        """ChromaDB 코사인 거리 기반 dense 검색. 반환: [(chunk_id, text, meta, score)]"""
        embedding = self.embeddings.embed_query(query)
        results = self.db._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs_list = results["documents"][0]
        metas_list = results["metadatas"][0]
        dists_list = results["distances"][0]

        # ChromaDB가 빈 문서를 None으로 반환하는 경우 대비
        filtered = [
            (doc, meta, dist)
            for doc, meta, dist in zip(docs_list, metas_list, dists_list)
            if doc is not None
        ]
        if not filtered:
            return []

        docs_f, metas_f, dists_f = zip(*filtered)
        # 거리가 작을수록 유사도 높음 → 정규화 후 반전
        dense_scores = [1.0 - s for s in minmax_norm(list(dists_f))]

        out: List[Tuple[str, str, Dict[str, Any], float]] = []
        for doc, md, sc in zip(docs_f, metas_f, dense_scores):
            md = dict(md or {})
            cid = md.get("chunk_id")
            if not cid:
                cid = (
                    f"{md.get('source','')}|{md.get('page','')}|"
                    f"{md.get('slide','')}|{md.get('sheet','')}|"
                    f"{md.get('row','')}"
                )
            out.append((cid, doc, md, sc))
        return out

    @traceable(name="Retriever.BM25Search")
    def bm25_search(
        self, query: str, k: int
    ) -> List[Tuple[str, str, Dict[str, Any], float]]:
        """BM25 키워드 검색. 반환: [(chunk_id, text, meta, score)]"""
        q_tokens = tokenize(query)
        scores = list(map(float, self.bm25.get_scores(q_tokens)))
        scores_norm = minmax_norm(scores)

        top_idx = sorted(
            range(len(scores_norm)), key=lambda i: scores_norm[i], reverse=True
        )[:k]

        out: List[Tuple[str, str, Dict[str, Any], float]] = []
        for i in top_idx:
            cid = self.bm25_chunk_ids[i]
            text = self.bm25_texts[i]
            md = dict(self.bm25_metas[i] or {})
            out.append((cid, text, md, scores_norm[i]))
        return out

    @traceable(name="Retriever.MergeScores")
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 20,
        bm25_k: int = 50,
        alpha: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """dense + BM25 결과를 chunk_id 기준으로 병합해서 score = alpha*dense + (1-alpha)*bm25 계산

        alpha: dense 비중(0~1). 키워드 검색 비중 높이려면 낮추기
        """
        dense = self.dense_search(query, dense_k)
        bm25 = self.bm25_search(query, bm25_k)

        merged: Dict[str, Dict[str, Any]] = {}

        for cid, text, md, sc in dense:
            merged[cid] = {
                "chunk_id": cid,
                "text": text,
                "metadata": md,
                "dense": sc,
                "bm25": 0.0,
                "score": sc,
            }

        for cid, text, md, sc in bm25:
            if cid in merged:
                merged[cid]["bm25"] = sc
            else:
                merged[cid] = {
                    "chunk_id": cid,
                    "text": text,
                    "metadata": md,
                    "dense": 0.0,
                    "bm25": sc,
                    "score": 0.0,
                }

        for _, item in merged.items():
            item["score"] = alpha * item["dense"] + (1.0 - alpha) * item["bm25"]

            md = item["metadata"]
            if "image_paths" in md:
                md["image_paths"] = safe_json_loads(md["image_paths"])

        ranked_all = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        return ranked_all[:top_k]


def _source_dedup_key(md: Dict[str, Any]) -> str:
    """같은 출처(파일+페이지/슬라이드/행)에서 온 청크를 하나로 취급하기 위한 키"""
    source = md.get("source", "")
    doc_type = (md.get("doc_type") or "").lower()
    if doc_type == "pdf":
        return f"{source}|p{md.get('page', '')}"
    if doc_type == "pptx":
        return f"{source}|s{md.get('slide', '')}"
    if doc_type == "xlsx":
        return f"{source}|{md.get('sheet', '')}|r{md.get('row', '')}"
    return f"{source}|{md.get('page', '')}|{md.get('slide', '')}|{md.get('sheet', '')}|{md.get('row', '')}"


def format_source(md: Dict[str, Any]) -> str:
    """metadata를 받아서 "파일명 p.3" 같은 출처 표시 문자열로 변환"""
    fn = md.get("file_name") or os.path.basename(md.get("source", ""))
    doc_type = (md.get("doc_type") or "").lower()

    if doc_type == "pdf" and md.get("page") is not None:
        return f"{fn} p.{md.get('page')}"
    if doc_type == "pptx" and md.get("slide") is not None:
        return f"{fn} slide {md.get('slide')}"
    if doc_type == "xlsx":
        sheet = md.get("sheet")
        row = md.get("row")
        if sheet is not None and row is not None:
            return f"{fn} [{sheet}] row {row}"
        if sheet is not None:
            return f"{fn} [{sheet}]"
    return f"{fn}"
