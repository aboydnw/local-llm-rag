# rag-lab

Local-first RAG framework. Point at any markdown docs corpus, get an answerable index. Built for learning, not for production.

Status: in development.

## Evaluating quality

rag-lab ships with an eval harness so you can measure changes rather than guess at them.

```bash
rag-lab eval --golden golden.yml --report report.md
rag-lab eval --golden golden.yml --report report.md --previous eval-reports/baseline.md  # diff
rag-lab eval --golden golden.yml --report report.md --judge  # opt-in Anthropic LLM judge
```

The harness scores retrieval (recall@k, MRR) and answer quality (keyword coverage, plus an
optional model-graded judge). The `--judge` flag needs the optional `judge` extra
(`uv pip install 'rag-lab[judge]'`) and an `ANTHROPIC_API_KEY`; everything else runs fully local.

`examples/devseed-oss/` holds a golden set covering titiler, eoAPI, and stac-fastapi.
`eval-reports/` is the project's eval trail — every report is committed so you can see how a
change moved the metrics.
