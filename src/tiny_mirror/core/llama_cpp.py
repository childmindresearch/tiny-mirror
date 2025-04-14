"""Tools for sending/receiving data from/to Llama.cpp."""

import functools
import json
from typing import Any, cast

import pydantic
import requests

from tiny_mirror.core import config

logger = config.get_logger()


class CompletionRequest(pydantic.BaseModel):
    """Request model for the /completion endpoint.

    This model contains all possible parameters for generating text completions.
    C.f. https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md
    for details.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    prompt: str | list[int | str] | list[str | list[int | str]] = pydantic.Field(
        ...,
        description=(
            "The prompt for text completion, can be a"
            " string, array of tokens, or mixed types."
        ),
    )
    temperature: float = pydantic.Field(
        0.8,
        description=(
            "Adjust the randomness of the generated t"
            "ext. Higher values increase randomness."
        ),
    )
    dynatemp_range: float = pydantic.Field(
        0.0,
        description=(
            "Dynamic temperature range. The final temperature"
            " will be in the range of [temperature - dynatemp_range; temperature + "
            "dynatemp_range]."
        ),
    )
    dynatemp_exponent: float = pydantic.Field(
        1.0, description="Dynamic temperature exponent."
    )
    top_k: int = pydantic.Field(
        40, description="Limit the next token selection to the K most probable tokens."
    )
    top_p: float = pydantic.Field(
        0.95,
        description=(
            "Limit the next token selection to a subset"
            "of tokens with a cumulative probability above a threshold P."
        ),
    )
    min_p: float = pydantic.Field(
        0.05,
        description=(
            "The minimum probability for a token to be"
            "considered, relative to the probability of the most likely token."
        ),
    )
    n_predict: int = pydantic.Field(
        -1, description="Maximum number of tokens to predict. -1 means infinity."
    )
    n_indent: int = pydantic.Field(
        0,
        description=(
            "Minimum line indentation for the generated"
            "text in number of whitespace characters."
        ),
    )
    n_keep: int = pydantic.Field(
        0,
        description=(
            "Number of tokens from the prompt to retain"
            "when context size is exceeded. -1 to retain all tokens."
        ),
    )
    stream: bool = pydantic.Field(
        default=False,
        description=(
            "Allows receiving each predicted token in"
            " real-time instead of waiting for the completion to finish."
        ),
    )
    stop: list[str] = pydantic.Field(
        [],
        description=(
            "Array of stopping strings. These words will"
            " not be included in the completion."
        ),
    )
    typical_p: float = pydantic.Field(
        1.0,
        description=(
            "Enable locally typical sampling with parameterp. 1.0 is disabled."
        ),
    )
    repeat_penalty: float = pydantic.Field(
        1.1,
        description=(
            "Control the repetition of token sequences in the generated text."
        ),
    )
    repeat_last_n: int = pydantic.Field(
        64,
        description=(
            "Last n tokens to consider for penalizing"
            " repetition. 0 is disabled, -1 is ctx-size."
        ),
    )
    presence_penalty: float = pydantic.Field(
        0.0, description="Repeat alpha presence penalty. 0.0 is disabled."
    )
    frequency_penalty: float = pydantic.Field(
        0.0, description="Repeat alpha frequency penalty. 0.0 is disabled."
    )
    dry_multiplier: float = pydantic.Field(
        0.0,
        description=(
            "DRY (Don't Repeat Yourself) repetition "
            "penalty multiplier. 0.0 is disabled."
        ),
    )
    dry_base: float = pydantic.Field(
        1.75, description="DRY repetition penalty base value."
    )
    dry_allowed_length: int = pydantic.Field(
        2,
        description=(
            "Tokens that extend repetition beyond this"
            " receive exponentially increasing penalty."
        ),
    )
    dry_penalty_last_n: int = pydantic.Field(
        -1,
        description=(
            "How many tokens to scan for repetitions."
            " 0 is disabled, -1 is context size."
        ),
    )
    dry_sequence_breakers: list[str] = pydantic.Field(
        ["\n", ":", '"', "*"],
        description="Array of sequence breakers for DRY sampling",
    )
    xtc_probability: float = pydantic.Field(
        0.0, description="Chance for token removal via XTC sampler. 0.0 is disabled."
    )
    xtc_threshold: float = pydantic.Field(
        0.1,
        description=(
            "Minimum probability threshold for tokens"
            " to be removed via XTC sampler (> 0.5 disables XTC)."
        ),
    )
    mirostat: int = pydantic.Field(
        0,
        description=(
            "Enable Mirostat sampling. 0 is disabled, 1 is Mirostat, 2 is Mirostat 2.0."
        ),
    )
    mirostat_tau: float = pydantic.Field(
        5.0, description="Mirostat target entropy, parameter tau."
    )
    mirostat_eta: float = pydantic.Field(
        0.1, description="Mirostat learning rate, parameter eta."
    )
    grammar: str | None = pydantic.Field(
        None, description="Grammar for grammar-based sampling."
    )
    json_schema: dict[str, Any] | None = pydantic.Field(
        None, description="JSON schema for grammar-based sampling."
    )
    seed: int = pydantic.Field(
        -1, description="Random number generator (RNG) seed. -1 for a random seed."
    )
    ignore_eos: bool = pydantic.Field(
        default=False, description="Ignore end of stream token and continue generating."
    )
    logit_bias: list[list[int | str | float | bool]] = pydantic.Field(
        [],
        description=(
            "Modify the likelihood of tokens appearing"
            " in the generated text completion."
        ),
    )
    n_probs: int = pydantic.Field(
        0,
        description=(
            "If > 0, include probabilities of top N tokens for each generated token."
        ),
    )
    min_keep: int = pydantic.Field(
        0, description="If > 0, force samplers to return N possible tokens at minimum."
    )
    t_max_predict_ms: int = pydantic.Field(
        0,
        description=(
            "Time limit in milliseconds for the prediction phase. 0 is disabled."
        ),
    )
    image_data: list[dict[str, str | int]] | None = pydantic.Field(
        None,
        description=(
            "Array of objects containing base64-encoded"
            " image data and IDs, for multimodal models."
        ),
    )
    id_slot: int = pydantic.Field(
        -1,
        description=(
            "Assign the completion task to a specific slot. -1 assigns to an idle slot."
        ),
    )
    cache_prompt: bool = pydantic.Field(
        default=True, description="Re-use KV cache from a previous request if possible."
    )
    return_tokens: bool = pydantic.Field(
        default=False,
        description="Return the raw generated token ids in the tokens field.",
    )
    samplers: list[str] = pydantic.Field(
        ["dry", "top_k", "tfs_z", "top_p", "min_p", "xtc", "temperature"],
        description="The order the samplers should be applied in.",
    )
    timings_per_token: bool = pydantic.Field(
        default=False,
        description=(
            "Include prompt processing and text generation"
            " speed information in each response."
        ),
    )
    post_sampling_probs: bool = pydantic.Field(
        default=False,
        description=(
            "Returns the probabilities of top n_probs"
            " tokens after applying sampling chain."
        ),
    )
    response_fields: list[str] | None = pydantic.Field(
        None, description="List of response fields to include in the response."
    )
    lora: list[dict[str, int | float]] | None = pydantic.Field(
        None,
        description=(
            "List of LoRA adapters to be applied to"
            "this specific request, with ID and scale fields."
        ),
    )


class LlmClient(pydantic.BaseModel):
    """A Llama CPP Large Language Model .

    Attributes:
        url: The URL of the Llama CPP server.
    """

    url: pydantic.HttpUrl = pydantic.Field(..., description="Llama.cpp URL")

    @property
    def _completion_endpoint(self) -> str:
        return str(self.url) + "completion"

    @property
    def _embedding_endpoint(self) -> str:
        return str(self.url) + "v1/embeddings"

    @staticmethod
    def prepare_prompt(system_prompt: str, user_prompt: str) -> str:
        """Formats a system and user prompt into a Llama prompt.

        Args:
            system_prompt: The instructions.
            user_prompt: Text to apply the instructions to.

        Returns:
            A formatted prompt.
        """
        return f"<system>{system_prompt}</system>\n\n<user>{user_prompt}</user>"

    def completion(self, request: CompletionRequest) -> str:
        """Runs a completion request via Llama.cpp.

        Args:
            request: The completion request to run.

        Returns:
            The generated text.
        """
        data = {
            key: value
            for key, value in request.model_dump().items()
            if value is not None
        }
        response = self.cached_post_request(
            self._completion_endpoint, payload=json.dumps(data), expected_codes=(200,)
        )

        return cast("str", response["content"])

    def embedding(self, text: str) -> tuple[float, ...]:
        """Runs a embedding request via Llama.cpp.

        Args:
            text: The text to embed.

        Returns:
            The embedding.
        """
        response = self.cached_post_request(
            self._embedding_endpoint,
            payload=json.dumps({"input": text}),
            expected_codes=(200,),
        )

        return tuple(response["data"][0]["embedding"])

    @staticmethod
    @functools.lru_cache(maxsize=2048)
    def cached_post_request(
        endpoint: str, payload: str, *, expected_codes: tuple[int] | None = None
    ) -> dict[str, Any]:
        """Runs a request and caches the response.

        Args:
            endpoint: The endpoint to send the request to.
            payload: The payload, must be in a format compatible with request.post()'s
                json argument and hashable.
            expected_codes: If not None, will error if the returned HTTP code is not in
                this list.

        Returns:
            The response.

        Raises:
            RuntimeError: If an unexpected HTTP code is returned.
        """
        with requests.Session() as session:
            response = session.post(endpoint, data=payload, timeout=60)
            if (
                expected_codes is not None
                and response.status_code not in expected_codes
            ):
                msg = (
                    f"Endpoint POST {endpoint} returned {response.status_code}, "
                    f"expected one of: {', '.join(str(code) for code in expected_codes)}."
                    f"Got message: {response.text} for payload {payload}."
                )
                raise RuntimeError(msg)

            return cast("dict[str, Any]", response.json())
