# Paper in Plain English

<!-- BEGIN: MCP usage (auto-generated, do not edit by hand) -->
## MCP usage

Explain an arXiv paper in plain English.

**MCP servers consumed:**
- **mcp-knowledge** — `get_arxiv_paper` · `search_arxiv`

**Inline `@tool` defs:** none — every tool comes from MCP.

<!-- END: MCP usage -->

Paste an arXiv link or ID (e.g. `1706.03762`) and the agent explains the paper for a
smart non-expert: the problem, the big idea, why it matters, and a glossary of its jargon.
It only uses what the abstract and metadata say. If you give a paper name, it searches arXiv first.

## Run

```bash
pip install -r requirements.txt
python main.py --port 28836   # then open http://127.0.0.1:28836
```

Set `LLM_PROVIDER`, `LLM_MODEL`, `AGENT_SETTING_CONFIG` and the matching API key first
(see the repo README's provider table).

## Example prompts

- `Explain arXiv 1706.03762 in plain English`
- `What is https://arxiv.org/abs/2005.14165 about?`
- `Explain the BERT paper simply`
- `Summarize arXiv 1512.03385 for a beginner`
- `What problem does 2106.09685 solve?`

## Tools

From the `knowledge` MCP server (no API key needed):

- `get_arxiv_paper` — fetch a paper's title, authors, abstract and links by arXiv ID
- `search_arxiv` — find papers by keyword when no ID is given
