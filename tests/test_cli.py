"""Tests for the CLI entry point and model-weights resolution.

Nothing here needs MAFFT, RNAduplex, a GPU or the network.
"""

from __future__ import annotations

import argparse
import json
import types
from unittest.mock import patch

import pytest

from lamp_design.cli import (
    EXIT_INPUT_ERROR,
    EXIT_USAGE,
    WEIGHTS_ENV_VAR,
    WeightsError,
    _register_command,
    build_parser,
    main,
    resolve_weights,
)

_PATTERN = "ACGTTGCAGGATCCTAGCATGCAAT"


def make_seq(length: int, shift: int = 0) -> str:
    offset = shift % len(_PATTERN)
    pattern = _PATTERN[offset:] + _PATTERN[:offset]
    return (pattern * (length // len(pattern) + 1))[:length]


@pytest.fixture
def target_fasta(tmp_path):
    """Three equal-length clean sequences: valid input, no aligner needed."""
    path = tmp_path / "targets.fa"
    path.write_text(
        "".join(f">s{i}\n{make_seq(400, shift=i)}\n" for i in range(3))
    )
    return path


# ---------------------------------------------------------------------------
# Parser surface
# ---------------------------------------------------------------------------


class TestCLI:
    def test_no_args_prints_help_and_exits_1(self):
        with patch("sys.argv", ["lamp-design"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == EXIT_USAGE

    def test_help(self):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_version(self):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0

    @pytest.mark.parametrize("command", ["prepare", "design", "evaluate"])
    def test_subcommand_help(self, command):
        with pytest.raises(SystemExit) as exc:
            main([command, "--help"])
        assert exc.value.code == 0, f"failed for subcommand: {command}"

    def test_subcommands_registered(self):
        parser = build_parser()
        actions = [
            a for a in parser._actions if hasattr(a, "choices") and a.choices
        ]
        assert set(actions[0].choices) == {"prepare", "design", "evaluate"}

    def test_unknown_command_is_a_usage_error(self):
        with pytest.raises(SystemExit) as exc:
            main(["nonexistent"])
        assert exc.value.code == 2

    def test_prepare_requires_in(self):
        with pytest.raises(SystemExit) as exc:
            main(["prepare"])
        assert exc.value.code == 2

    def test_top_level_help_mentions_the_weights_env_var(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        assert WEIGHTS_ENV_VAR in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_prepare_writes_a_panel(self, target_fasta, tmp_path, capsys):
        out = tmp_path / "panel.json"

        main(["prepare", "--in", str(target_fasta), "--out", str(out)])

        data = json.loads(out.read_text())
        assert data["n_targets"] == 3
        assert data["budget"] == 3 * 400
        assert len(data["consensus"]) == 400
        assert set(data["consensus"]) <= set("ACGT")
        assert "Prepared 3 target sequence(s)" in capsys.readouterr().out
        # The funnel is written next to the output, PrimerCast-style.
        assert (tmp_path / "panel.json.diagnostics.json").exists()

    def test_prepare_without_out_is_a_qc_only_run(self, target_fasta, capsys, tmp_path):
        main(["prepare", "--in", str(target_fasta)])
        assert "Prepared 3 target sequence(s)" in capsys.readouterr().out
        assert list(tmp_path.glob("*.json")) == []

    def test_prepare_writes_consensus_and_alignment_fasta(
        self, target_fasta, tmp_path
    ):
        consensus = tmp_path / "consensus.fa"
        alignment = tmp_path / "alignment.fa"

        main(
            [
                "prepare",
                "--in", str(target_fasta),
                "--consensus-out", str(consensus),
                "--alignment-out", str(alignment),
            ]
        )

        assert consensus.read_text().startswith(">consensus\n")
        assert alignment.read_text().count(">") == 3

    def test_bad_input_exits_2_with_a_message_on_stderr(self, tmp_path, capsys):
        bad = tmp_path / "gapped.fa"
        seq = make_seq(300)
        bad.write_text(f">hbv\n{seq[:100]}---{seq[100:]}\n")

        with pytest.raises(SystemExit) as exc:
            main(["prepare", "--in", str(bad)])

        assert exc.value.code == EXIT_INPUT_ERROR
        err = capsys.readouterr().err
        assert err.startswith("error: ")
        # Names the sequence, the position, and what to do -- but no longer
        # offers --pre-aligned, which is no longer a way past a gap.
        assert "'hbv'" in err and "position 101" in err
        assert "not supported" in err and "--pre-aligned" not in err

    def test_over_budget_exits_2(self, tmp_path, capsys):
        path = tmp_path / "big.fa"
        path.write_text("".join(f">s{i}\n{make_seq(1000, shift=i)}\n" for i in range(4)))

        with pytest.raises(SystemExit) as exc:
            main(["prepare", "--in", str(path), "--max-budget", "2000"])

        assert exc.value.code == EXIT_INPUT_ERROR
        assert "over the cap of 2,000" in capsys.readouterr().err

    def test_missing_input_file_exits_2(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["prepare", "--in", str(tmp_path / "nope.fa")])
        assert exc.value.code == EXIT_INPUT_ERROR
        assert "error: " in capsys.readouterr().err

    def test_every_command_is_real(self):
        """No stubs remain, and each subcommand dispatches to its own module.

        The stub machinery this replaces existed while the pipeline was being
        built; `evaluate` was the last placeholder. If a command module ever
        goes missing, `build_parser` should now raise on import rather than
        quietly registering something that refuses to run.
        """
        from lamp_design.commands import design, evaluate, prepare

        parser = build_parser()
        action = next(
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
        )
        expected = {"prepare": prepare, "design": design, "evaluate": evaluate}
        assert set(action.choices) == set(expected)
        for name, module in expected.items():
            subparser = action.choices[name]
            assert subparser.get_default("func") is module.run, name
            assert "NOT IMPLEMENTED" not in (subparser.description or ""), name

    def test_a_missing_command_module_is_not_swallowed(self, monkeypatch):
        """A broken command must surface, not degrade into a usable-looking CLI."""
        import importlib

        real = importlib.import_module

        def fail(name, *args, **kwargs):
            if name == "lamp_design.commands.evaluate":
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fail)
        with pytest.raises(ModuleNotFoundError):
            build_parser()

    @pytest.mark.parametrize("command", ["design", "evaluate"])
    def test_bad_weights_path_is_a_user_error(self, command, tmp_path, capsys):
        # A weights typo is a user error, so it exits 2 with a message rather
        # than a traceback -- from the real command and from a stub alike.
        argv = {
            "design": ["design", "--in", "panel.json", "--out", "sets.tsv"],
            "evaluate": [
                "evaluate", "--in", "primers.tsv", "--targets", "panel.json",
                "--out", "s.tsv",
            ],
        }[command]
        with pytest.raises(SystemExit) as exc:
            main(argv + ["--weights", str(tmp_path / "nope")])

        assert exc.value.code == EXIT_INPUT_ERROR
        err = capsys.readouterr().err
        assert "does not exist (from --weights)" in err


# ---------------------------------------------------------------------------
# Weights resolution
# ---------------------------------------------------------------------------


class TestWeightsResolution:
    def test_explicit_path_wins(self, tmp_path):
        other = tmp_path / "env"
        other.mkdir()
        assert resolve_weights(str(tmp_path), env={WEIGHTS_ENV_VAR: str(other)}) == tmp_path

    def test_env_var_is_the_fallback(self, tmp_path):
        assert resolve_weights(None, env={WEIGHTS_ENV_VAR: str(tmp_path)}) == tmp_path

    def test_empty_explicit_value_falls_back_to_the_env_var(self, tmp_path):
        assert resolve_weights("", env={WEIGHTS_ENV_VAR: str(tmp_path)}) == tmp_path

    def test_reads_the_real_environment_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv(WEIGHTS_ENV_VAR, str(tmp_path))
        assert resolve_weights() == tmp_path

    def test_unset_falls_back_to_the_bundled_ensemble(self):
        """A fresh clone runs with no flags and no environment."""
        from lamp_design.models.inference import BUNDLED_WEIGHTS

        assert resolve_weights(None, env={}) == BUNDLED_WEIGHTS

    def test_unset_still_resolves_when_not_required(self):
        from lamp_design.models.inference import BUNDLED_WEIGHTS

        assert resolve_weights(None, env={}, required=False) == BUNDLED_WEIGHTS

    def test_missing_bundled_weights_is_an_error_naming_both_ways_to_set_it(
        self, monkeypatch, tmp_path
    ):
        """The only way to reach the error now is a checkout that lost them."""
        monkeypatch.setattr(
            "lamp_design.models.inference.BUNDLED_WEIGHTS", tmp_path / "gone"
        )
        with pytest.raises(WeightsError) as exc:
            resolve_weights(None, env={})
        message = str(exc.value)
        assert "--weights" in message
        assert WEIGHTS_ENV_VAR in message
        assert "bundled" in message

    def test_missing_bundled_weights_returns_none_when_not_required(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            "lamp_design.models.inference.BUNDLED_WEIGHTS", tmp_path / "gone"
        )
        assert resolve_weights(None, env={}, required=False) is None

    def test_configured_but_missing_path_is_always_an_error(self, tmp_path):
        missing = tmp_path / "checkpoints"
        # A typo is an error even in the non-required mode: it is a mistake,
        # not an absent setting.
        with pytest.raises(WeightsError) as exc:
            resolve_weights(None, env={WEIGHTS_ENV_VAR: str(missing)}, required=False)
        assert str(missing) in str(exc.value)
        assert f"${WEIGHTS_ENV_VAR}" in str(exc.value), "names where the value came from"

    def test_error_names_the_weights_flag_as_the_source(self, tmp_path):
        with pytest.raises(WeightsError, match="--weights"):
            resolve_weights(str(tmp_path / "nope"), env={})

    def test_user_home_is_expanded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "ckpt").mkdir()
        assert resolve_weights("~/ckpt", env={}) == tmp_path / "ckpt"


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------


class TestCommandRegistration:
    def test_a_command_module_is_handed_the_subparsers(self, monkeypatch):
        """Registration is by module name, so a new command needs no edit here."""
        registered = []
        fake = types.SimpleNamespace(register=lambda sub: registered.append(sub))
        monkeypatch.setattr(
            "importlib.import_module",
            lambda path: fake if path == "lamp_design.commands.design" else pytest.fail(path),
        )

        parser = build_parser.__globals__["argparse"].ArgumentParser()
        subparsers = parser.add_subparsers()
        _register_command(subparsers, "design")

        assert registered == [subparsers]

    def test_a_missing_dependency_surfaces(self, monkeypatch):
        """A command whose own imports are broken must fail loudly.

        There used to be a stub fallback that a ModuleNotFoundError could hide
        behind; registration is now unconditional, so this just has to stay
        true.
        """

        def boom(path):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")

        monkeypatch.setattr("importlib.import_module", boom)

        with pytest.raises(ModuleNotFoundError, match="torch"):
            _register_command(None, "design")
