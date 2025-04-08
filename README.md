# Tiny-Mirror

## Example Usage

### Concerning Summary
There are two manners of getting the concerning summary. The first uses the `run()` method which provides 
the final result only. The alternative is to use the other public methods of the `ConcerningSummary` 
class to get all the intermediate results.

```python
from tiny_mirror.procedures.summaries import concerning
from tiny_mirror.core import llama_cpp

# Set up the basics
entry = "I have been feeling super anxious about the world today."
llama_cpp_url = "http://localhost:8001"
client = llama_cpp.LlmClient(url=llama_cpp_url) 
writer = concerning.ConcerningSummary(client)

# Get only the final result
result = writer.run(entry)

# Get intermediate results
summary = writer.create_summary(entry)
summary_no_think_block = summary[summary.find("/think>") + 9 :]
reflection_1, reflection_2, reframe, support = summary_no_think_block.split(
    "\n"
)

reframe_similarities = writer.find_similar(reframe, concerning.REFRAME_TEMPLATES)
best_reframe = reframe_similarities[0].sentence
support_similarities = writer.find_similar(support, concerning.SUPPORT_TEMPLATES)
best_support = support_similarities[0].sentence

all_results = {
    "summary": f"{reflection_1}\n{reflection_2}\n{best_reframe}\n{best_support}",
    "raw_summary": summary,
    "reframe_similarities": reframe_similarities,
    "support_similarities": support_similarities,
}
```