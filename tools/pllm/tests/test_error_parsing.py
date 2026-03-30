"""
Unit tests for error-parsing helpers in ollama_helper_tester.py.

Two test groups:
  1. parse_could_not_find_version_error  -- pure regex, no LLM needed, always runs.
  2. process_error / OllamaHelper        -- requires langchain; skipped if not installed.
  3. get_python_range                    -- requires PyPIQuery helpers; skipped if deps missing.

Run with:
    python3 tools/pllm/tests/test_error_parsing.py
  or (if pytest installed):
    python3 -m pytest tools/pllm/tests/test_error_parsing.py -v
"""
import sys
import os
import unittest

# Allow running from repo root or from tools/pllm/
_THIS = os.path.dirname(os.path.abspath(__file__))
_PLLM = os.path.dirname(_THIS)
if _PLLM not in sys.path:
    sys.path.insert(0, _PLLM)

# ---------------------------------------------------------------------------
# Attempt to import the module under test; note which parts are available.
# ---------------------------------------------------------------------------
_PARSE_AVAILABLE = False
_HELPER_AVAILABLE = False
_PYPI_AVAILABLE = False

try:
    # parse_could_not_find_version_error uses only stdlib (re).
    # Extract just that function before any langchain imports run.
    import types
    with open(os.path.join(_PLLM, "helpers", "ollama_helper_tester.py")) as fh:
        src = fh.read()
    fn_start = src.find("def parse_could_not_find_version_error")
    fn_end = src.find("\nfrom helpers.py_pi_query", fn_start)
    fn_end = fn_end if fn_end != -1 else src.find("\nfrom helpers.ollama_helper_base", fn_start)
    fn_src = "import re\n" + (src[fn_start:fn_end] if fn_end != -1 else src[fn_start:])
    _mod = types.ModuleType("_cnf_only")
    exec(compile(fn_src, "ollama_helper_tester.py", "exec"), _mod.__dict__)
    parse_could_not_find_version_error = _mod.parse_could_not_find_version_error
    _PARSE_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Could not load parse_could_not_find_version_error: {e}")

try:
    from helpers.ollama_helper_tester import OllamaHelper, parse_could_not_find_version_error as _pcnf
    parse_could_not_find_version_error = _pcnf  # prefer full import if available
    _HELPER_AVAILABLE = True
    _PARSE_AVAILABLE = True
except ImportError:
    pass  # langchain not installed, skip those tests

try:
    from helpers.py_pi_query import PyPIQuery
    _PYPI_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Group 1 — parse_could_not_find_version_error (always runs if extraction ok)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_PARSE_AVAILABLE, "parse_could_not_find_version_error not importable")
class TestParseCNFVersionError(unittest.TestCase):

    def test_full_form_with_available_versions(self):
        error = (
            '{"stream":"ERROR: Could not find a version that satisfies the requirement '
            'djangorestframework==3.15.2 (from versions: 3.14.0, 3.15.0, 3.15.1)\\n"}'
        )
        result = parse_could_not_find_version_error(error)
        self.assertIsNotNone(result)
        self.assertEqual(result["package"], "djangorestframework")
        self.assertEqual(result["requested_version"], "3.15.2")
        self.assertIn("3.14.0", result["available_versions"])
        self.assertIn("3.15.0", result["available_versions"])
        self.assertIn("3.15.1", result["available_versions"])

    def test_full_form_no_parens(self):
        error = (
            "ERROR: Could not find a version that satisfies the requirement "
            "requests==99.0.0"
        )
        result = parse_could_not_find_version_error(error)
        self.assertIsNotNone(result)
        self.assertEqual(result["package"], "requests")
        self.assertEqual(result["requested_version"], "99.0.0")
        self.assertIsInstance(result["available_versions"], list)

    def test_no_matching_distribution_form(self):
        error = "ERROR: No matching distribution found for numpy==0.0.1"
        result = parse_could_not_find_version_error(error)
        self.assertIsNotNone(result)
        self.assertEqual(result["package"], "numpy")
        self.assertEqual(result["requested_version"], "0.0.1")

    def test_unrelated_error_returns_none(self):
        error = "Traceback (most recent call last):\n  ModuleNotFoundError: No module named 'foo'"
        result = parse_could_not_find_version_error(error)
        self.assertIsNone(result)

    def test_hyphenated_package_name(self):
        error = (
            "ERROR: Could not find a version that satisfies the requirement "
            "python-dateutil==99.0 (from versions: 2.8.2, 2.9.0)"
        )
        result = parse_could_not_find_version_error(error)
        self.assertIsNotNone(result)
        self.assertEqual(result["package"], "python-dateutil")
        self.assertEqual(result["requested_version"], "99.0")
        self.assertIn("2.8.2", result["available_versions"])

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_could_not_find_version_error(""))

    def test_no_matching_distribution_standalone(self):
        """Standalone 'No matching distribution found' (without 'Could not find a version')."""
        error = "ERROR: No matching distribution found for flask==0.0.1"
        result = parse_could_not_find_version_error(error)
        self.assertIsNotNone(result)
        self.assertEqual(result["package"], "flask")


# ---------------------------------------------------------------------------
# Group 2 — process_error dispatch (requires langchain / full helper import)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HELPER_AVAILABLE, "langchain_community not installed, skipping OllamaHelper tests")
class TestProcessErrorClassification(unittest.TestCase):

    def _make_helper(self):
        class StubModel:
            def invoke(self, *a, **kw):
                return {}

        class StubPyPI:
            def check_module_name(self, names):
                return names if isinstance(names, list) else [names]
            def read_module_file(self, *a, **kw):
                return ""

        helper = object.__new__(OllamaHelper)
        helper.logging = False
        helper.rag = False
        helper.pypi = StubPyPI()
        helper.model = StubModel()
        helper.could_not_find_version = lambda *a, **kw: {"module": "pkg", "version": "1.0"}
        helper.dependency_conflict = lambda *a, **kw: {"module": "pkg", "version": "1.0"}
        helper.import_error = lambda *a, **kw: {"module": "pkg", "version": "1.0"}
        helper.module_not_found = lambda *a, **kw: {"module": "pkg", "version": "1.0"}
        helper.attribute_error = lambda *a, **kw: {"module": "pkg", "version": "1.0"}
        helper.invalid_version = lambda *a, **kw: {"module": "pkg", "version": "1.0"}
        helper.non_zero_error = lambda *a, **kw: "pkg"
        helper.non_zero_error_version = lambda *a, **kw: {"module": "pkg", "version": "1.0"}
        helper.syntax_error_helper = lambda *a, **kw: {"module": "pkg", "version": "1.0"}
        return helper

    def test_could_not_find_version_dispatch(self):
        h = self._make_helper()
        _, etype = h.process_error(
            "ERROR: Could not find a version that satisfies the requirement foo==1.2",
            {}, {}
        )
        self.assertEqual(etype, "VersionNotFound")

    def test_no_matching_distribution_dispatch(self):
        """'No matching distribution found' alone should map to VersionNotFound."""
        h = self._make_helper()
        _, etype = h.process_error(
            "ERROR: No matching distribution found for bar==9.9",
            {}, {}
        )
        self.assertEqual(etype, "VersionNotFound")

    def test_pips_dependency_resolver_dispatch(self):
        h = self._make_helper()
        _, etype = h.process_error(
            "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.",
            {}, {}
        )
        self.assertEqual(etype, "DependencyConflict")

    def test_resolution_impossible_dispatch(self):
        h = self._make_helper()
        _, etype = h.process_error(
            "pip._internal.exceptions.DistributionNotFound: ResolutionImpossible: for help visit ...",
            {}, {}
        )
        self.assertEqual(etype, "DependencyConflict")

    def test_dependency_conflicts_dispatch(self):
        h = self._make_helper()
        _, etype = h.process_error(
            "ERROR: package-a 1.0 has requirement package-b>=2.0, but you have package-b 1.0 which causes dependency conflicts.",
            {}, {}
        )
        self.assertEqual(etype, "DependencyConflict")

    def test_import_error_dispatch(self):
        h = self._make_helper()
        _, etype = h.process_error("ImportError: cannot import name 'foo' from 'bar'", {}, {})
        self.assertEqual(etype, "ImportError")

    def test_module_not_found_dispatch(self):
        h = self._make_helper()
        _, etype = h.process_error("ModuleNotFoundError: No module named 'scipy'", {}, {})
        self.assertEqual(etype, "ModuleNotFound")

    def test_non_zero_code_dispatch(self):
        h = self._make_helper()
        _, etype = h.process_error("returned a non-zero code: 1", {}, {})
        self.assertEqual(etype, "NonZeroCode")

    def test_no_error_returns_none_signal(self):
        """Clean run output (no error keywords) → error_type='None' (success signal)."""
        h = self._make_helper()
        _, etype = h.process_error("", {}, {})
        self.assertEqual(etype, "None")

    def test_syntax_error_dispatch(self):
        h = self._make_helper()
        _, etype = h.process_error("SyntaxError: invalid syntax", {}, {})
        self.assertEqual(etype, "SyntaxError")


# ---------------------------------------------------------------------------
# Group 3 — get_python_range correctness (requires PyPIQuery)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_PYPI_AVAILABLE, "PyPIQuery not importable (missing deps), skipping range tests")
class TestGetPythonRange(unittest.TestCase):

    def _make_pypi(self):
        return PyPIQuery(logging=False, base_modules="/tmp/pllm_test_modules")

    def test_range_zero_preserves_detected_version(self):
        pypi = self._make_pypi()
        versions = pypi.get_python_range("3.8", pyrange=0)
        self.assertEqual(len(versions), 1, f"Expected exactly 1 version for pyrange=0, got {versions}")
        self.assertEqual(versions[0], "3.8", f"Expected '3.8', got {versions}")

    def test_range_zero_does_not_return_27_for_python3_versions(self):
        pypi = self._make_pypi()
        for ver in ["3.7", "3.9", "3.10", "3.11"]:
            versions = pypi.get_python_range(ver, pyrange=0)
            self.assertNotEqual(
                versions[0], "2.7",
                f"get_python_range({ver!r}, 0) wrongly returned 2.7: {versions}"
            )

    def test_range_one_returns_three_versions(self):
        pypi = self._make_pypi()
        versions = pypi.get_python_range("3.9", pyrange=1)
        self.assertEqual(len(versions), 3, f"Expected 3 versions for pyrange=1, got {versions}")

    def test_range_one_does_not_silently_replace_with_27(self):
        pypi = self._make_pypi()
        versions = pypi.get_python_range("3.9", pyrange=1)
        self.assertNotIn("2.7", versions, f"2.7 unexpectedly found in {versions}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
