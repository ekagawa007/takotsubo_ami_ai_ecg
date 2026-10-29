"""Exercise notebook state guards without Colab, a GPU, or network access."""

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "colab_inference.ipynb"


def notebook_cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def cell_source(index):
    return "".join(notebook_cells()[index]["source"])


def load_functions(index, namespace):
    """Load the actual guard definitions, without running the setup cell."""
    parsed = ast.parse(cell_source(index))
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)


def execute_without_imports(index, namespace):
    """Use mocked globals, stopping at an intentional early setup failure."""
    class RemoveImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return ast.copy_location(ast.Pass(), node)

        def visit_ImportFrom(self, node):
            return ast.copy_location(ast.Pass(), node)

    parsed = ast.fix_missing_locations(RemoveImports().visit(ast.parse(cell_source(index))))
    exec(compile(parsed, str(NOTEBOOK), "exec"), namespace)


def fail_guard():
    raise RuntimeError("Intentional upstream validation failure")


class NotebookIntegrityTests(unittest.TestCase):
    def test_every_code_cell_compiles(self):
        for index, cell in enumerate(notebook_cells()):
            if cell["cell_type"] == "code":
                with self.subTest(cell=index):
                    compile("".join(cell["source"]), f"colab cell {index}", "exec")

    def test_published_notebook_contains_no_execution_records(self):
        for index, cell in enumerate(notebook_cells()):
            if cell["cell_type"] == "code":
                with self.subTest(cell=index):
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs"), [])


class NotebookRerunTests(unittest.TestCase):
    def test_failed_source_selection_invalidates_previous_success(self):
        state = {name: True for name in (
            "ENVIRONMENT_READY", "WEIGHTS_READY", "DEVICE_READY", "INPUT_READY", "OPTIONS_READY")}
        state.update(result={"predictions": []}, input_record={"old": True}, IMAGE_PATH=Path("old.png"),
                     PREPARED_IMAGE_PATH=Path("old.png"),
                     RESULT_INPUT_TOKEN="old-input", RESULT_OPTIONS_SIGNATURE="old-options",
                     sys=SimpleNamespace(version_info=(3, 10)))
        with self.assertRaises(RuntimeError):
            execute_without_imports(2, state)
        for name in ("ENVIRONMENT_READY", "WEIGHTS_READY", "DEVICE_READY", "INPUT_READY", "OPTIONS_READY"):
            self.assertFalse(state[name], name)
        for name in ("result", "input_record", "IMAGE_PATH", "PREPARED_IMAGE_PATH", "RESULT_INPUT_TOKEN", "RESULT_OPTIONS_SIGNATURE"):
            self.assertIsNone(state[name], name)

    def test_failed_input_preparation_does_not_reuse_old_image_or_prediction(self):
        state = dict(INPUT_READY=True, OPTIONS_READY=True, IMAGE_PATH=Path("old.png"),
                     PREPARED_IMAGE_PATH=Path("old.png"),
                     input_record={"old": True}, INPUT_TOKEN="old-input", result={"predictions": []},
                     RESULT_INPUT_TOKEN="old-input", RESULT_OPTIONS_SIGNATURE="old-options",
                     check_execution_device=fail_guard)
        with self.assertRaisesRegex(RuntimeError, "Intentional"):
            execute_without_imports(10, state)
        self.assertFalse(state["INPUT_READY"])
        self.assertFalse(state["OPTIONS_READY"])
        for name in ("IMAGE_PATH", "PREPARED_IMAGE_PATH", "input_record", "INPUT_TOKEN", "result", "RESULT_INPUT_TOKEN", "RESULT_OPTIONS_SIGNATURE"):
            self.assertIsNone(state[name], name)

    def test_failed_options_rerun_invalidates_previous_prediction(self):
        state = dict(OPTIONS_READY=True, result={"predictions": []}, RESULT_OPTIONS_SIGNATURE="old-options",
                     check_prepared_input=fail_guard)
        with self.assertRaisesRegex(RuntimeError, "Intentional"):
            execute_without_imports(12, state)
        self.assertFalse(state["OPTIONS_READY"])
        self.assertIsNone(state["result"])
        self.assertIsNone(state["RESULT_OPTIONS_SIGNATURE"])

    def test_failed_inference_rerun_invalidates_previous_prediction(self):
        state = dict(result={"predictions": []}, RESULT_INPUT_TOKEN="old-input",
                     RESULT_OPTIONS_SIGNATURE="old-options", check_prepared_options=fail_guard)
        with self.assertRaisesRegex(RuntimeError, "Intentional"):
            execute_without_imports(15, state)
        for name in ("result", "RESULT_INPUT_TOKEN", "RESULT_OPTIONS_SIGNATURE"):
            self.assertIsNone(state[name], name)

    def test_display_and_download_refuse_stale_results_before_accessing_files(self):
        for index in (18, 20, 21):
            with self.subTest(cell=index):
                state = {"check_current_result": fail_guard, "result": {"predictions": []}}
                with self.assertRaisesRegex(RuntimeError, "Intentional"):
                    execute_without_imports(index, state)


class NotebookPreparedInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="ecg_notebook_", ignore_cleanup_errors=True)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "input.png"
        self.source.write_bytes(b"unchanged test bytes")
        self.state = {
            "check_execution_device": lambda: None, "hashlib": hashlib,
            "INPUT_READY": True, "input_record": {"input_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest()},
            "INPUT_SOURCE": "Google Drive", "USER_IMAGE_PATH": str(self.source),
            "IMAGE_PREPARATION": "Require 480 x 480", "INPUT_POLICY": "strict480",
            "IMAGE_PATH": self.source, "PREPARED_IMAGE_PATH": self.source,
        }
        self.state["INPUT_SELECTION"] = tuple(self.state[key] for key in (
            "INPUT_SOURCE", "USER_IMAGE_PATH", "IMAGE_PREPARATION", "INPUT_POLICY"))
        load_functions(10, self.state)

    def test_unchanged_prepared_input_is_ready(self):
        self.assertIsNone(self.state["check_prepared_input"]())

    def test_input_selection_and_policy_changes_require_preparation_again(self):
        replacements = {"INPUT_SOURCE": "Synthetic example", "USER_IMAGE_PATH": "another.png",
                        "IMAGE_PREPARATION": "White padding, then resize", "INPUT_POLICY": "pad"}
        for name, value in replacements.items():
            original = self.state[name]
            with self.subTest(field=name):
                self.state[name] = value
                with self.assertRaises(RuntimeError):
                    self.state["check_prepared_input"]()
                self.state[name] = original

    def test_missing_or_changed_input_file_blocks_inference(self):
        self.source.write_bytes(b"changed after preparation")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.state["check_prepared_input"]()
        self.source.unlink()
        with self.assertRaises(RuntimeError):
            self.state["check_prepared_input"]()

    def test_replacing_input_path_requires_preparation_even_if_bytes_are_equal(self):
        replacement = self.root / "another.png"
        replacement.write_bytes(self.source.read_bytes())
        self.state["IMAGE_PATH"] = replacement
        with self.assertRaises(RuntimeError):
            self.state["check_prepared_input"]()

    def test_unready_state_or_missing_record_blocks_inference(self):
        self.state["INPUT_READY"] = False
        with self.assertRaises(RuntimeError):
            self.state["check_prepared_input"]()
        self.state["INPUT_READY"] = True
        self.state["input_record"] = None
        with self.assertRaises(RuntimeError):
            self.state["check_prepared_input"]()


class NotebookPreparedOptionsTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "json": json, "check_prepared_input": lambda: None,
            "MODEL": "ensemble", "DEVICE": "cuda", "PRECISION": "float32", "INPUT_POLICY": "strict480",
            "TTA_PRESET": "Paper defaults", "CUSTOM_VIT_VIEWS": 24, "CUSTOM_VIT_MAGNITUDE": 1,
            "CUSTOM_CONVNEXTV2_VIEWS": 1, "CUSTOM_CONVNEXTV2_MAGNITUDE": 5,
            "CUSTOM_MLPMIXER_VIEWS": 32, "CUSTOM_MLPMIXER_MAGNITUDE": 3,
            "VIT_TTA": 24, "VIT_MAGNITUDE": 1, "CONVNEXTV2_TTA": 1, "CONVNEXTV2_MAGNITUDE": 5,
            "MLPMIXER_TTA": 32, "MLPMIXER_MAGNITUDE": 3,
            "ENSEMBLE_WEIGHTS": [1.0, 1.0, 1.0], "THRESHOLD": None, "SEED": 1234,
            "BATCH_SIZE": 1, "NUM_WORKERS": 4, "ENABLE_GRADCAM": True,
            "CAM_TARGET": "model-predicted", "CAM_NORMALIZATION": "inference", "CAM_OPACITY": 0.6,
            "OPTIONS_READY": True, "INPUT_TOKEN": "current-input", "result": {"predictions": []},
            "RESULT_INPUT_TOKEN": "current-input",
        }
        load_functions(12, self.state)
        self.state["OPTIONS_SIGNATURE"] = self.state["current_options_signature"]()
        self.state["RESULT_OPTIONS_SIGNATURE"] = self.state["OPTIONS_SIGNATURE"]

    def test_unchanged_options_and_linked_result_are_ready(self):
        self.assertIsNone(self.state["check_prepared_options"]())
        self.assertIsNone(self.state["check_current_result"]())

    def test_changed_tta_device_and_other_options_block_inference(self):
        replacements = {
            "TTA_PRESET": "No TTA", "CUSTOM_VIT_VIEWS": 2, "VIT_TTA": 2, "VIT_MAGNITUDE": 2,
            "DEVICE": "cpu", "NUM_WORKERS": 0, "SEED": 12, "INPUT_POLICY": "pad",
            "ENABLE_GRADCAM": False, "CAM_OPACITY": 0.2, "THRESHOLD": 0.4,
            "ENSEMBLE_WEIGHTS": [2.0, 1.0, 1.0],
        }
        for name, value in replacements.items():
            original = copy.deepcopy(self.state[name])
            with self.subTest(field=name):
                self.state[name] = value
                with self.assertRaises(RuntimeError):
                    self.state["check_prepared_options"]()
                self.state[name] = original

    def test_unprepared_options_block_inference(self):
        self.state["OPTIONS_READY"] = False
        with self.assertRaises(RuntimeError):
            self.state["check_prepared_options"]()

    def test_old_input_token_or_options_signature_cannot_be_displayed(self):
        for field in ("RESULT_INPUT_TOKEN", "RESULT_OPTIONS_SIGNATURE"):
            original = self.state[field]
            with self.subTest(field=field):
                self.state[field] = "previous-run"
                with self.assertRaises(RuntimeError):
                    self.state["check_current_result"]()
                self.state[field] = original

    def test_missing_prediction_cannot_be_displayed(self):
        self.state["result"] = None
        with self.assertRaises(RuntimeError):
            self.state["check_current_result"]()


if __name__ == "__main__":
    unittest.main()
