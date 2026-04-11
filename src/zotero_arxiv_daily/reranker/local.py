from .base import BaseReranker, register_reranker
import logging
import warnings
import numpy as np


@register_reranker("local")
class LocalReranker(BaseReranker):
    def __init__(self, config):
        super().__init__(config)

        from sentence_transformers import SentenceTransformer

        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging

            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        model_name = self.config.reranker.local.model
        trust_remote_code = getattr(self.config.reranker.local, "trust_remote_code", False)
        self.encoder = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        encode_kwargs = self.config.reranker.local.encode_kwargs or {}
        s1_feature = self.encoder.encode(s1, **encode_kwargs, show_progress_bar=True)
        s2_feature = self.encoder.encode(s2, **encode_kwargs, show_progress_bar=True)
        sim = self.encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()
