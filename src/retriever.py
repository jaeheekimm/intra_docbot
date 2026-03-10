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

# ★ Kiwi는 한 번만 생성(매 호출마다 만들면 느림)
_KIWI = Kiwi()


def _import_chroma():
    # langchain-chroma가 있으면 그걸 우선 사용
    try:
        from langchain_chroma import Chroma  # type: ignore

        return Chroma
    except Exception:
        from langchain_community.vectorstores import Chroma  # type: ignore

        return Chroma


# ★ bm25_index.py와 반드시 동일하게
def tokenize(text: str) -> List[str]:
    tokens = []
    for tok in _KIWI.tokenize(text):
        if tok.tag.startswith("N") or tok.tag.startswith("V"):
            tokens.append(tok.form)
    return tokens


def minmax_norm(scores: List[float]) -> List[float]:
    """BM25 점수를 실제 값 기반으로 0~1 정규화. 점수 차이를 그대로 반영."""
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-9:
        return [1.0 if s == max_s else 0.0 for s in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


def rank_norm(scores: List[float], reverse: bool = True) -> List[float]:
    """
    순위 기반 정규화.
    - reverse=True: 점수가 클수록 좋음
    - reverse=False: 점수가 작을수록 좋음 (distance)
    결과는 0~1 사이 값.
    """
    if not scores:
        return scores

    indexed = list(enumerate(scores))

    # 정렬 기준
    if reverse:
        sorted_idx = sorted(indexed, key=lambda x: x[1], reverse=True)
    else:
        sorted_idx = sorted(indexed, key=lambda x: x[1])

    n = len(scores)
    out = [0.0] * n

    for rank, (orig_i, _) in enumerate(sorted_idx):
        # 1등 = 1.0, 마지막 = 0.0
        if n == 1:
            out[orig_i] = 1.0
        else:
            out[orig_i] = 1.0 - (rank / (n - 1))

    return out


def safe_json_loads(x: Any) -> Any:
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

        # langchain_chroma 버전이면 persist_directory, community 버전이면 persist_directory가 맞음(둘 다 대응)
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

        embedding = self.embeddings.embed_query(query)
        results = self.db._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs_list = results["documents"][0]
        metas_list = results["metadatas"][0]
        dists_list = results["distances"][0]

        # None page_content 필터링 (ChromaDB가 빈 문서를 None으로 반환하는 경우 대비)
        filtered = [
            (doc, meta, dist)
            for doc, meta, dist in zip(docs_list, metas_list, dists_list)
            if doc is not None
        ]
        if not filtered:
            return []

        docs_f, metas_f, dists_f = zip(*filtered)
        # 거리가 작을수록 유사도 높음 → minmax 후 반전
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
        """
        alpha: dense 비중(0~1). 키워드가 더 중요하면 낮추기.
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
    """
    같은 출처(파일+페이지/슬라이드/행)에서 온 청크를 하나로 취급하기 위한 키.
    hybrid_search에서 동일 페이지의 중복 청크를 제거할 때 사용.
    """
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
