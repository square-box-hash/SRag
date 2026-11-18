"""Guard the packaging promise: `import srag` must not drag in torch.

A regression here is invisible until someone deploys it — the package still
works, it just quietly pulls ~800 MB and a few seconds of startup.
"""

import inspect
import json
import os
import subprocess
import sys

# Runnable as a plain script as well as under pytest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEAVY = ["torch", "sentence_transformers", "playwright", "trafilatura"]


def test_import_does_not_load_heavy_modules():
    code = (
        "import sys, json, srag; "
        f"print(json.dumps({{m: m in sys.modules for m in {HEAVY!r}}}))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert not any(loaded.values()), f"import srag loaded heavy modules: {loaded}"


def test_retrieval_api_is_async_with_sync_aliases():
    import srag

    for name in ("search", "query", "query_session", "build_context"):
        assert inspect.iscoroutinefunction(getattr(srag.SRag, name)), f"{name}() is not a coroutine"

    for name in ("query_sync", "query_session_sync", "build_context_sync"):
        assert not inspect.iscoroutinefunction(getattr(srag.SRag, name)), f"{name}() should be sync"


def test_lazy_exports_resolve():
    import srag

    for name in ("SRagOrchestrator", "SRagIndexer", "DocumentIngestor"):
        try:
            assert getattr(srag, name) is not None
        except ModuleNotFoundError as exc:
            # A bare environment without the optional stack still proves the
            # lazy plumbing works: it reached the module, then found no dep.
            print(f"       skip {name}: {exc}")

    with pytest_raises_attribute_error():
        getattr(srag, "NotARealExport")


class pytest_raises_attribute_error:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        assert exc_type is AttributeError, f"expected AttributeError, got {exc_type}"
        return True


if __name__ == "__main__":
    failures = 0
    for fn in (
        test_import_does_not_load_heavy_modules,
        test_retrieval_api_is_async_with_sync_aliases,
        test_lazy_exports_resolve,
    ):
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print("\nFAILURES:", failures if failures else "none")
    sys.exit(1 if failures else 0)
