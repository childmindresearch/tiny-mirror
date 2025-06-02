"""Runs concerning summaries for Mirror."""

import dataclasses
import math
import pathlib
from collections.abc import Sequence

from tiny_mirror.core import llama_cpp

# c.f. tiny_mirror.core.llama_cpp.CompletionRequest for allowed keys and values.
COMPLETION_PARAMETERS = {
    "dry_multiplier": 1.0,
    "logit_bias": [
        [13, 1],  # Increase likelihood of ".".
        [323, False],  # Disable " and".
        [477, False],  # Disable " or".
    ],
}

DATA_DIR = pathlib.Path(__file__).parent / "data"
SYSTEM_PROMPT = (DATA_DIR / "prompt.txt").read_text()
REFRAME_TEMPLATES = (DATA_DIR / "reframe_templates.txt").read_text().split("\n")
SUPPORT_TEMPLATES = (DATA_DIR / "support_templates.txt").read_text().split("\n")
GRAMMAR = (DATA_DIR / "prompt.gbnf").read_text()
RERANK_LLAMA_ARG_UBATCH = 2048  # Physical batch size of the rerank server.


@dataclasses.dataclass
class Similarity:
    """Binds template sentences to a similarity score."""

    sentence: str
    similarity: float

    def __lt__(self, other: "Similarity") -> bool:
        """Implements "less than" for sorting."""
        return self.similarity < other.similarity


class ConcerningSummary:
    """Creates summaries for concerning entries."""

    def __init__(
        self,
        llm_client: llama_cpp.LlmClient,
        embedding_client: llama_cpp.LlmClient,
        rerank_client: llama_cpp.LlmClient,
    ) -> None:
        """Initializes the ConcerningSummary object.

        Args:
            llm_client: The LLM llama cpp client to use.
            embedding_client: The embedding llama cpp client to use.
            rerank_client: The rerank llama cpp client to use.
        """
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.rerank_client = rerank_client

    def run(self, entry: str, n_reframe_reranks: int = 5) -> str:
        """Runs the full concerning summary pipeline.

        This is intended as the production use-case. If you want to extract
        more granular data like template probabilities, use the other public
        methods instead.

        Note that we only use as many characters as the reranker accepts tokens.
        This could likely be far higher, but represents a safeguard against the most
        extreme case wherein every character is a token.

        Args:
            entry: The journal entry to summarize.
            n_reframe_reranks: The number of templates to consider in reranking reframe
             templates.

        Returns:
            The summary.
        """
        summary = self.create_summary(entry)
        summary_no_think_block = summary[summary.find("/think>") + 9 :]
        reflection_1, reflection_2, reframe, support = summary_no_think_block.split(
            "\n"
        )

        reframe_similarities = self.embed(reframe, REFRAME_TEMPLATES)
        support_similarities = self.embed(support, SUPPORT_TEMPLATES)
        best_support = support_similarities[0].sentence

        best_reframe_lines = [
            reframe.sentence for reframe in reframe_similarities[:n_reframe_reranks]
        ]
        rerank_response = self.rerank(
            entry,
            best_reframe_lines,
            clip_needle=True,
            clip_haystack=False,
        )
        best_reframe = best_reframe_lines[rerank_response.results[0].index]

        return f"{reflection_1}\n{reflection_2}\n{best_reframe}\n{best_support}"

    def create_summary(self, entry: str) -> str:
        """Creates the entry's summary.

        Args:
            entry: The journal entry to summarize.

        Returns:
            The summary.
        """
        prompt = self.llm_client.prepare_prompt(
            system_prompt=SYSTEM_PROMPT, user_prompt=entry
        )
        request = llama_cpp.CompletionRequest(
            prompt=prompt, grammar=GRAMMAR, **COMPLETION_PARAMETERS
        )

        return self.llm_client.completion(request=request)

    def embed(
        self, needle: str, haystack: Sequence[str], *, preserve_order: bool = False
    ) -> tuple[Similarity, ...]:
        """Uses embeddings to find strings similar to the needle.

        The embedding used is the default of the Llama.cpp server.

        Args:
            needle: The string to find similar strings to.
            haystack: The strings to compare the needle to.
            preserve_order: If True, the output will be in the same
                order as the haystack. If false, the output will be
                sorted by similarity to the needle, descending.

        Returns:
            The similarity scores of the haystack.
        """
        needle_embedding = self.embedding_client.embedding(needle)

        similarities = [
            Similarity(
                sentence=sentence,
                similarity=_cosine_similarity(
                    needle_embedding, self.embedding_client.embedding(sentence)
                ),
            )
            for sentence in haystack
        ]

        if preserve_order:
            return tuple(similarities)
        return tuple(sorted(similarities, reverse=True))

    def rerank(
        self,
        needle: str,
        haystack: Sequence[str],
        *,
        clip_needle: bool = False,
        clip_haystack: bool = False,
        clip_size: int = RERANK_LLAMA_ARG_UBATCH,
    ) -> llama_cpp.RerankResponse:
        """Reranks the haystack.

        Clipping may be necessary to prevent hitting the llama.cpp server's physical
        batch size limit.

        Args:
            needle: The string to find similar strings to.
            haystack: The strings to compare the needle to.
            clip_needle: If true, clips the needle to the requested size.
            clip_haystack: If true, clips each haystack element to the requested size.
            clip_size: The size to clip to in tokens.

        Returns:
            The response from llama.cpp
        """
        if clip_needle:
            needle = self.clip(self.rerank_client, needle, clip_size)

        if clip_haystack:
            haystack = [
                self.clip(self.rerank_client, hay, clip_size) for hay in haystack
            ]

        return self.rerank_client.rerank(needle, haystack)

    @staticmethod
    def clip(client: llama_cpp.LlmClient, content: str, size: int) -> str:
        """Clips the content to a specified number of tokens.

        Args:
            client: The llama.cpp client to tokenize with.
            content: The string to clip.
            size: The size to clip to in tokens.

        Returns:
            The clipped string.
        """
        response = client.tokenize(content, with_pieces=True)
        return "".join([token.piece for token in response.tokens[:size]])


def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Calculate the cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity.

    Raises:
        ValueError: If either vector has magnitude zero
    """
    dot_product = sum(x * y for x, y in zip(vec1, vec2, strict=True))
    magnitude1 = math.sqrt(sum(x * x for x in vec1))
    magnitude2 = math.sqrt(sum(y * y for y in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        msg = "Cannot compute cosine similarity for zero vectors"
        raise ValueError(msg)

    return dot_product / (magnitude1 * magnitude2)
