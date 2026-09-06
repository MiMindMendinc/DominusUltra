"""Exercise the notebook's checkout code with real local Git repositories."""

import ast
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "colab/DominusUltra_GPU_Evidence.ipynb"


def load_checkout():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    tree = ast.parse(source)
    # Load the real helper without running Colab setup or installing dependencies.
    tree.body = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.FunctionDef))
    ]
    namespace = {}
    exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    return namespace["checkout_commit"]


def git(directory, *args):
    return subprocess.check_output(
        ["git", *args], cwd=directory, text=True, stderr=subprocess.PIPE
    ).strip()


class ColabCheckoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote"
        self.remote.mkdir()
        git(self.remote, "init", "-b", "main")
        git(self.remote, "config", "user.name", "Checkout test")
        git(self.remote, "config", "user.email", "checkout@example.invalid")
        self.target = self.commit("reviewed kernel\n")
        self.latest = self.commit("later kernel\n")
        self.checkout = self.root / "checkout"
        self.checkout_commit = load_checkout()

    def commit(self, content):
        (self.remote / "kernel.py").write_text(content, encoding="utf-8")
        git(self.remote, "add", "kernel.py")
        git(self.remote, "commit", "-m", "Update fixture")
        return git(self.remote, "rev-parse", "HEAD")

    def run_checkout(self, target=None):
        return self.checkout_commit(
            str(self.remote), str(self.checkout), target or self.target
        )

    def test_pins_reviewed_commit_when_remote_branch_has_moved(self):
        self.assertNotEqual(self.target, self.latest)
        self.assertEqual(self.run_checkout(), self.target)
        self.assertEqual(git(self.checkout, "rev-parse", "HEAD"), self.target)
        self.assertEqual(git(self.checkout, "branch", "--show-current"), "")
        self.assertEqual(git(self.checkout, "status", "--porcelain"), "")
        self.assertEqual(
            (self.checkout / "kernel.py").read_text(encoding="utf-8"), "reviewed kernel\n"
        )

    def test_repeated_setup_keeps_same_commit_after_another_remote_change(self):
        self.run_checkout()
        self.commit("another unreviewed change\n")
        self.assertEqual(self.run_checkout(), self.target)
        self.assertEqual(git(self.checkout, "rev-parse", "HEAD"), self.target)

    def test_dirty_checkout_stops_without_discarding_local_work(self):
        self.run_checkout()
        (self.checkout / "kernel.py").write_text("local edit\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Save local changes"):
            self.run_checkout(self.latest)
        self.assertEqual(git(self.checkout, "rev-parse", "HEAD"), self.target)
        self.assertEqual(
            (self.checkout / "kernel.py").read_text(encoding="utf-8"), "local edit\n"
        )

    def test_untracked_work_stops_checkout(self):
        self.run_checkout()
        note = self.checkout / "operator-note.txt"
        note.write_text("keep this\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Save local changes"):
            self.run_checkout(self.latest)
        self.assertEqual(note.read_text(encoding="utf-8"), "keep this\n")

    def test_branch_names_and_abbreviated_shas_are_rejected_before_clone(self):
        for target in ("main", "agent/reproducible-gpu-evidence", self.target[:7]):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "40-character"):
                    self.run_checkout(target)
                self.assertFalse(self.checkout.exists())

    def test_missing_commit_stops_before_changing_existing_head(self):
        self.run_checkout()
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_checkout("0" * 40)
        self.assertEqual(git(self.checkout, "rev-parse", "HEAD"), self.target)


if __name__ == "__main__":
    unittest.main()
