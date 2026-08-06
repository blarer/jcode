# jcode repo notes

## code-review-graph

This repo has a code knowledge graph with local embeddings, kept fresh by the
`code-review-graph` watch daemon. Query it instead of blind grep when the
question is structural.

| Question | Tool |
| --- | --- |
| What is this repo, how risky is my change? | `get_minimal_context_tool(task=...)`, call first, ~100 tokens |
| Review my diff | `detect_changes_tool(base="master", detail_level="minimal")` |
| What breaks if I change X? | `get_impact_radius_tool` |
| Who calls or is called by X? | `query_graph_tool(pattern="callers_of" / "callees_of" / "tests_for")` |
| Where is the code that does Y? | `semantic_search_nodes_tool` |
| Architecture overview | `get_architecture_overview_tool` |
| Rename a symbol safely | `refactor_tool(mode="rename")` then `apply_refactor_tool` |

Prefer `detail_level="minimal"`. Plain grep is still better for exact string or
comment matches. `refactor_tool(mode="dead_code")` exceeds the MCP timeout on a
repo this size; run it through the CLI instead.
