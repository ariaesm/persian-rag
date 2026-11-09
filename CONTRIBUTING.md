# Contributing

Thanks for helping improve persian-rag! Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/ariaesm/persian-rag.git
cd persian-rag
python -m venv .venv && source .venv/bin/activate
make install   # pip install -e ".[dev]"
make check     # ruff + pytest, same as CI
```

## Guidelines

- Keep runtime dependencies minimal: the standard library and numpy only.
  Optional integrations (e.g. pypdf) must be imported lazily.
- Add tests for every change; network calls must be mocked.
- Use real Persian examples in normalizer/tokenizer tests, and explain any
  linguistic rule in a comment.
- Code must pass `ruff check` and `ruff format --check`, with type hints and
  docstrings on public functions.
- Sample corpus texts must be original writing (no copied articles).
