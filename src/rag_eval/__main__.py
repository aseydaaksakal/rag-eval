"""``python -m rag_eval`` is the same as the ``rag-eval`` command."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
