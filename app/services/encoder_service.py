"""Skill extraction with a local transformer encoder: BERT, RoBERTa or DistilBERT.

This is the fourth backend, and the only one that is both free per query and offline.
It exists so the comparison has a point that is neither a hosted API nor a managed vector
database, and so the obvious question has an answer: is a 110M-parameter encoder running
on your own machine enough for this task?

How it works. The skill catalogue is embedded once at startup. A query is embedded the
same way, and the nearest entries by cosine similarity are returned. Embeddings are the
mean of the last hidden states over the attention mask, which is the standard pooling for
models that were not trained with a sentence objective: taking the [CLS] token of a plain
BERT gives a vector that was never optimised to mean anything on its own.

What it cannot do. Like the vector-database backend, it is bounded by the catalogue, so
it will never return a skill nobody wrote down. Unlike a generative model, it also cannot
invent one. The benchmark is what shows which of those two properties matters more.
"""

import logging
import re
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Three encoders of the same family and roughly the same training recipe, which is what
# makes comparing them informative: differences come from pre-training, not architecture.
MODELS = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
}

DEFAULT_CATALOGUE = Path(__file__).resolve().parent.parent.parent / "data" / "skills.txt"


class EncoderService:
    """Loads one encoder and the skill catalogue, then answers by nearest neighbour."""

    def __init__(self, settings=None, model_key: str | None = None,
                 catalogue_path: str | Path | None = None, batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_key = model_key or os.getenv("ENCODER_MODEL", "bert")
        if self.model_key not in MODELS:
            raise ValueError(f"unknown encoder {self.model_key!r}, expected one of {list(MODELS)}")
        self.model_name = MODELS[self.model_key]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

        path = Path(catalogue_path or os.getenv("SKILL_CATALOGUE", DEFAULT_CATALOGUE))
        self.skills = self._load_catalogue(path)
        self.batch_size = batch_size
        self.skill_vectors = self._embed(self.skills)

        logger.info("EncoderService ready: %s on %s, %d skills in the catalogue",
                    self.model_name, self.device, len(self.skills))

    @staticmethod
    def _load_catalogue(path: Path) -> List[str]:
        if not path.exists():
            raise FileNotFoundError(f"skill catalogue not found at {path}")
        with open(path, encoding="utf-8") as fh:
            skills = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
        if not skills:
            raise ValueError(f"skill catalogue at {path} is empty")
        return skills

    def _embed(self, texts: List[str]):
        """Mean-pool the last hidden state over the attention mask, then L2-normalise.

        Normalising here means the similarity below is a plain dot product, so the query
        path does no division per candidate.
        """
        torch = self._torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=256, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        return torch.cat(out, dim=0)

    @staticmethod
    def _spans(query: str, max_len: int = 3) -> List[str]:
        """Every 1 to 3 word window of the query, deduplicated and order-preserving.

        This exists because of a mistake worth recording. Embedding a whole job advert
        into one vector and comparing it to single-word skills compares objects of
        completely different lengths, and the nearest neighbours come back dominated by
        whatever the document is broadly about. Matching short windows instead compares
        like with like, which is what an encoder with no sentence objective can do well.
        """
        words = re.findall(r"[A-Za-z0-9#+./-]+", query)
        seen, spans = set(), []
        for n in range(1, max_len + 1):
            for i in range(len(words) - n + 1):
                span = " ".join(words[i:i + n]).lower()
                if span not in seen:
                    seen.add(span)
                    spans.append(span)
        return spans

    def search(self, query: str, top_k: int = 15, min_score: float = 0.0,
               mode: str = "spans") -> List[dict]:
        """Return the catalogue entries this query is closest to, with their similarity.

        `mode="spans"` scores every short window of the query against the catalogue and
        keeps, for each skill, its best-matching window. `mode="document"` embeds the
        whole query as one vector, which is the naive approach and is kept so the
        benchmark can show how much the difference is worth.

        The score is returned rather than hidden: thresholding it is what turns a fixed
        top-k list into a decision, and the benchmark sweeps that threshold.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        torch = self._torch

        if mode == "document":
            scores = (self._embed([query]) @ self.skill_vectors.T).squeeze(0)
        elif mode == "spans":
            spans = self._spans(query)
            if not spans:
                return []
            # (spans x skills), then each skill keeps its best window.
            scores = (self._embed(spans) @ self.skill_vectors.T).max(dim=0).values
        else:
            raise ValueError(f"unknown mode {mode!r}, expected 'spans' or 'document'")

        k = min(top_k, scores.shape[0])
        top = torch.topk(scores, k=k)
        return [
            {"skill": self.skills[idx], "score": round(float(score), 4)}
            for score, idx in zip(top.values.tolist(), top.indices.tolist())
            if score >= min_score
        ]
