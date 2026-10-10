"""Cache keys that invalidate themselves.

A hand-maintained VERSION constant is a poor cache key: it only changes when
someone remembers to change it. Two deploys of the macro work shipped correct
code that the app then served stale, because the computation changed and the
constant did not — and on the second one the constant *was* bumped, yet the
entry survived anyway. Hashing the source removes the human step entirely:
any edit to a named file changes the key, with nothing to remember.

Pass the result at the CALL SITE, never as a default argument::

    state = _macro_state_cached(source_version("macro.py"))

Defaults bind at definition time from whatever the name resolves to then, so
``def f(version=VERSION)`` looks parameterised but is not — the call site is
where the key must be computed if it is to be read fresh.

Verified by experiment rather than reasoning: with the app running, editing a
threshold in gold_lens.py and reloading the browser flipped the affected chip
and flipped back on revert, with no process restart. So Streamlit does reload
the local module — the code was never the stale part, only the key was. An
earlier version of this docstring blamed a stale ``sys.modules`` entry; that
explanation was wrong and the test above is what disproved it.
"""

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def source_version(*filenames: str) -> str:
    """A short digest of the given source files, read fresh on every call.

    Any edit to a named file changes the digest, so the cache entry keyed on
    it is dropped without anyone remembering to bump a constant. A file that
    cannot be read contributes its name instead of its bytes, so a missing
    file still produces a stable key rather than raising inside a cached
    function.
    """
    h = hashlib.sha256()
    for name in sorted(filenames):
        try:
            h.update((_ROOT / name).read_bytes())
        except OSError:
            h.update(name.encode("utf-8"))
    return h.hexdigest()[:16]
