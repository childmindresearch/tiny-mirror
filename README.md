# Tiny-Mirror

## Example Usage

### Concerning Summary
To run TinyMirror's concerning summaries end-to-end, use the following script.
If you want to extract data from each individual step, have a look at replicating
the `run` function in your environment.

```python
from tiny_mirror.procedures.summaries import concerning
from tiny_mirror.core import llama_cpp

# Set up the basics
entry = "I have been feeling super anxious about the world today."
embedding_client = llama_cpp.LlmClient(url="http://localhost:8000")
llm_client = llama_cpp.LlmClient(url="http://localhost:8001")
rerank_client = llama_cpp.LlmClient(url="http://localhost:8002")

writer = concerning.ConcerningSummary(llm_client, embedding_client, rerank_client)
result = writer.run(entry)
```