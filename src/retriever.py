import os
import json
import pickle
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from kiwipiepy import Kiwi
from langsmith import traceable

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./indexes/chroma_db")
COLLECTION = os.getenv("CHROMA_COLLECTION", "intra_docs")
BM25_PATH = os.getenv("BM25_PATH", "./indexes/bm25_index.pkl")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

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
                f"{bm25_path} 없음. python src/pipeline/bm25_index.py 먼저 실행해."
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

        hits = self.db.similarity_search_with_score(query, k=k)

        docs = [d for d, _ in hits]
        raw = [float(s) for _, s in hits]

        # similarity_search_with_score는 보통 distance 반환 (작을수록 좋음)
        dense_scores = rank_norm(raw, reverse=False)

        out: List[Tuple[str, str, Dict[str, Any], float]] = []
        for d, sc in zip(docs, dense_scores):
            md = dict(d.metadata or {})
            cid = md.get("chunk_id")
            if not cid:
                cid = (
                    f"{md.get('source','')}|{md.get('page','')}|"
                    f"{md.get('slide','')}|{md.get('sheet','')}|"
                    f"{md.get('row','')}"
                )
            out.append((cid, d.page_content, md, sc))
        return out

    @traceable(name="Retriever.BM25Search")
    def bm25_search(
        self, query: str, k: int
    ) -> List[Tuple[str, str, Dict[str, Any], float]]:
        q_tokens = tokenize(query)
        scores = list(map(float, self.bm25.get_scores(q_tokens)))
        scores_norm = rank_norm(scores, reverse=True)

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

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return ranked


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
