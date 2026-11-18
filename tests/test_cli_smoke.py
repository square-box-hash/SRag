"""CLI smoke tests.

Both of these regressions shipped green: `srag query` printed a coroutine after
`query()` became async in 1.3.0, and `srag search` / `srag index` raised
"'SRagResult' object is not subscriptable" after 1.3.4 started returning objects
instead of dicts. Nothing in the test suite invoked the CLI, so neither was caught.

No models, no network: SRag is replaced with a stub.
"""

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import srag  # noqa: E402
from srag import SRagResult  # noqa: E402
from srag.cli import main as cli_main  # noqa: E402


class FakeSRag:
    """Stands in for SRag so the CLI never loads a model."""

    used_query_sync = False

    def __init__(self, *args, **kwargs):
        pass

    def query_sync(self, query, session=None, k=5, debug=False):
        FakeSRag.used_query_sync = True
        return [
            {
                "content": "Moyna results: BJP holds the seat.",
                "url": "https://example.com/moyna",
                "title": "Moyna",
                "score": 0.72,
            }
        ]

    def query(self, *args, **kwargs):  # must NOT be called by the CLI
        raise AssertionError("CLI called the async query() without awaiting it")


def run_cli(argv):
    """Invoke main() with a patched argv and return its stdout."""
    original_srag = srag.SRag
    original_argv = sys.argv
    srag.SRag = FakeSRag
    sys.argv = ["srag"] + argv
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            cli_main()
    finally:
        srag.SRag = original_srag
        sys.argv = original_argv
    return buffer.getvalue()


def test_query_command_uses_the_sync_api():
    out = run_cli(["query", "who won", "--session", "web_test"])
    assert FakeSRag.used_query_sync, "srag query did not call query_sync()"
    assert "coroutine" not in out.lower(), "srag query printed a coroutine object"
    assert "Moyna" in out, out[:200]


def test_query_command_json_output_is_serialisable():
    out = run_cli(["query", "who won", "--session", "web_test", "--json"])
    assert "Moyna" in out
    assert "coroutine" not in out.lower()


def test_printers_accept_both_dict_and_sragresult():
    """Locks the 1.3.4 regression: printers must read either shape."""
    from srag.cli import _print_index_result, _print_ingest_result

    as_object = SRagResult(success=True, doc_count=3, chunk_count=9, indexed_count=8, topic="sports")
    as_dict = {"success": True, "doc_count": 3, "chunk_count": 9, "indexed_count": 8, "topic": "sports"}

    for payload in (as_object, as_dict):
        for printer in (_print_index_result, _print_ingest_result):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                try:
                    printer(payload, "web_test")
                except TypeError as exc:
                    raise AssertionError(f"{printer.__name__} rejected {type(payload).__name__}: {exc}")
            assert "8" in buffer.getvalue()


def test_failure_result_also_prints():
    from srag.cli import _print_index_result

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _print_index_result(SRagResult(success=False, reason="no_documents"), "web_test")
    assert "failed" in buffer.getvalue().lower()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")
    print("\nFAILURES:", failures if failures else "none")
    sys.exit(1 if failures else 0)
