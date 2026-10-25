"""Hermetic tests for deployment previews and shell dependency argument handling.

No test executes the deployment apply path. Parser/plan tests source functions;
preview tests replace SSH and rsync with harmless stderr-only executables.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "scripts/funding_arb_research/deploy/sync.sh"
BASH = shutil.which("bash") or "/bin/bash"


class PublicMaintenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="zabor maintenance ")
        self.addCleanup(self.temp.cleanup)
        self.scratch = Path(self.temp.name)
        self.bin_dir = self.scratch / "bin"
        self.bin_dir.mkdir()
        for name in ("ssh", "rsync"):
            stub = self.bin_dir / name
            stub.write_text("#!/bin/sh\nprintf 'REMOTE_COMMAND_WAS_CALLED\\n' >&2\nexit 91\n")
            stub.chmod(0o755)
        self.env = dict(os.environ)
        self.env["PATH"] = f"{self.bin_dir}{os.pathsep}{os.defpath}"
        self.env["VPS"] = "operator@preview-only.invalid"

    def run_bash(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [BASH, *args],
            cwd=self.scratch,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def parse_only(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self.run_bash(
            "-c",
            'source "$1"; shift; sync_parse_options "$@" || exit "$?"; '
            'printf "%s\\n%s\\n" "$SYNC_MODE" "$SYNC_ACTION"',
            "parse-only",
            str(SYNC),
            *args,
        )

    def assert_preview(self, *args: str) -> str:
        result = self.run_bash(str(SYNC), *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No commands executed", result.stdout)
        self.assertNotIn("REMOTE_COMMAND_WAS_CALLED", result.stdout + result.stderr)
        return result.stdout

    def test_default_is_offline_preview_even_with_target_configured(self) -> None:
        output = self.assert_preview()
        self.assertIn("[preview]", output)
        self.assertNotIn("systemctl", output)

    def test_install_without_apply_only_previews_the_install_plan(self) -> None:
        self.assertIn("install_vps.sh", self.assert_preview("--install"))

    def test_restart_without_apply_only_previews_the_restart_plan(self) -> None:
        self.assertIn("systemctl", self.assert_preview("--restart"))

    def test_explicit_preview_does_not_call_remote_programs(self) -> None:
        self.assert_preview("--preview", "--restart")

    def test_unknown_preview_option_stops_before_any_remote_program(self) -> None:
        result = self.run_bash(str(SYNC), "--preview", "--unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown flag", result.stderr)
        self.assertNotIn("REMOTE_COMMAND_WAS_CALLED", result.stdout + result.stderr)

    def test_sourcing_defines_functions_without_running_main(self) -> None:
        self.env.pop("VPS")
        result = self.run_bash("-c", 'source "$1"', "source-only", str(SYNC))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout + result.stderr, "")

    def test_help_needs_no_target_and_calls_no_remote_program(self) -> None:
        self.env.pop("VPS")
        result = self.run_bash(str(SYNC), "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--apply", result.stdout)
        self.assertNotIn("REMOTE_COMMAND_WAS_CALLED", result.stdout + result.stderr)

    def test_apply_install_options_parse_without_executing_a_plan(self) -> None:
        result = self.parse_only("--apply", "--install")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "apply\ninstall\n")

    def test_apply_unknown_option_is_rejected_by_pure_parser(self) -> None:
        result = self.parse_only("--apply", "--unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown flag", result.stderr)

    def test_conflicting_actions_are_rejected_by_pure_parser(self) -> None:
        self.assertEqual(self.parse_only("--install", "--restart").returncode, 2)

    def test_conflicting_modes_are_rejected_by_pure_parser(self) -> None:
        self.assertEqual(self.parse_only("--preview", "--apply").returncode, 2)

    def test_target_validation_in_plan_builder_is_not_a_network_probe(self) -> None:
        self.env["VPS"] = "bad host"
        result = self.run_bash(
            "-c",
            'source "$1"; sync_parse_options; sync_build_plan "$VPS"',
            "plan-only",
            str(SYNC),
        )
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("REMOTE_COMMAND_WAS_CALLED", result.stdout + result.stderr)

    def test_transfer_plan_protects_runtime_state_without_deletion(self) -> None:
        result = self.run_bash(
            "-c",
            'source "$1"; sync_parse_options; sync_build_plan "$VPS"; '
            'printf "%s\\0" "${SYNC_TRANSFER[@]}"',
            "plan-only",
            str(SYNC),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.rstrip("\0").split("\0")
        excludes = [args[i + 1] for i, arg in enumerate(args) if arg == "--exclude"]
        for name in (
            ".env", ".env.*", "*.env", "data/", "outputs/", "logs/",
            ".git", ".venv*", ".claude/", ".codex/", "CLAUDE.md", "AGENTS.md",
        ):
            self.assertIn(name, excludes)
        self.assertFalse(any(arg.startswith("--delete") for arg in args))
        self.assertEqual(args[-2], f"{SYNC.parent.parent}/")
        self.assertEqual(args[-1], "operator@preview-only.invalid:/opt/funding_arb/")
        self.assertNotIn("REMOTE_COMMAND_WAS_CALLED", result.stdout + result.stderr)

    def test_ci_passes_constraints_to_pip_instead_of_shell_redirection(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        section = workflow.split("- name: Install lint tooling\n", 1)[1]
        run_section = section.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0]
        command = "\n".join(line.removeprefix("          ") for line in run_section.splitlines())
        calls_file = self.scratch / "pip-arguments.jsonl"
        python_stub = self.bin_dir / "python"
        python_stub.write_text(
            f"#!{sys.executable}\n"
            "import json, sys\n"
            f"with open({str(calls_file)!r}, 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        )
        python_stub.chmod(0o755)
        result = self.run_bash("-c", command)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in calls_file.read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        requirements = calls[1][3:]
        for name in (
            "pydantic>=2.7", "python-dotenv>=1.0", "httpx>=0.27", "tenacity>=8.3",
            "PyYAML>=6.0", "pandas>=2.2", "numpy>=1.26", "rich>=13.7",
            "mypy>=1.10", "ruff>=0.6",
        ):
            self.assertIn(name, requirements)
        self.assertEqual(list(self.scratch.glob("=*")), [])


if __name__ == "__main__":
    unittest.main()
