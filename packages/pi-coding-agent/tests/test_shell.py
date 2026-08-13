"""Tests for shell helpers shared by the bash tool."""

from __future__ import annotations

import os
import subprocess

from pi_coding_agent.utils.shell import (
    _tracked_detached_child_pids,
    get_shell_env,
    kill_process_tree,
    kill_tracked_detached_children,
    sanitize_binary_output,
    track_detached_child_pid,
    untrack_detached_child_pid,
)


def test_get_shell_env_returns_a_copy(monkeypatch):
    monkeypatch.setenv("PI_SHELL_TEST", "value")

    env = get_shell_env("/tmp/pi-bin")

    assert env["PI_SHELL_TEST"] == "value"
    env["PI_SHELL_TEST"] = "changed"
    assert os.environ["PI_SHELL_TEST"] == "value"


def test_get_shell_env_prepends_the_managed_bin_directory(monkeypatch):
    """A bash command must find the `rg`/`fd` that `ensure_tool` downloaded."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    env = get_shell_env("/tmp/pi-bin")

    assert env["PATH"] == f"/tmp/pi-bin{os.pathsep}/usr/bin:/bin"


def test_get_shell_env_does_not_add_the_bin_directory_twice(monkeypatch):
    monkeypatch.setenv("PATH", f"/tmp/pi-bin{os.pathsep}/usr/bin")

    env = get_shell_env("/tmp/pi-bin")

    assert env["PATH"] == f"/tmp/pi-bin{os.pathsep}/usr/bin"


def test_get_shell_env_reuses_the_existing_path_key_casing(monkeypatch):
    """Windows spells it `Path`; a second `PATH` key would shadow the real one."""
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.setenv("Path", "/usr/bin")

    env = get_shell_env("/tmp/pi-bin")

    assert env["Path"] == f"/tmp/pi-bin{os.pathsep}/usr/bin"
    assert "PATH" not in env


def test_sanitize_binary_output_preserves_allowed_line_control_characters():
    text = "tab\tnewline\ncarriage\rreturn"

    assert sanitize_binary_output(text) == text


def test_sanitize_binary_output_strips_other_control_characters_and_del():
    assert sanitize_binary_output("a\x00b\x01c\x7fd") == "abcd"


def test_sanitize_binary_output_strips_lone_surrogates_and_format_characters():
    assert sanitize_binary_output("a\udc80b\u200dc\ufeffd") == "abcd"


def test_sanitize_binary_output_leaves_plain_text_unchanged():
    text = "plain text café"

    assert sanitize_binary_output(text) == text


def test_kill_tracked_detached_children_kills_the_whole_process_group():
    """Port of the `trackDetachedChildPid` / `killTrackedDetachedChildren` pair in
    `packages/coding-agent/src/utils/shell.ts`.

    The bash tool spawns children in their own session, so they never receive
    the terminal's SIGHUP. Without the tracked-pid set, quitting mid-command
    orphans the child (and any grandchild it spawned).
    """
    proc = subprocess.Popen(
        ["/bin/bash", "-c", "sleep 60 & sleep 60"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        track_detached_child_pid(proc.pid)

        kill_tracked_detached_children()

        assert proc.wait(timeout=5) is not None
        # The set is emptied so a second shutdown signal does not signal
        # already-reaped pids.
        assert proc.pid not in _tracked_detached_child_pids
    finally:
        untrack_detached_child_pid(proc.pid)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_untrack_detached_child_pid_prevents_a_later_kill():
    proc = subprocess.Popen(
        ["/bin/bash", "-c", "sleep 60"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        track_detached_child_pid(proc.pid)
        untrack_detached_child_pid(proc.pid)

        kill_tracked_detached_children()

        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_untrack_detached_child_pid_is_idempotent():
    untrack_detached_child_pid(-12345)
    untrack_detached_child_pid(-12345)


def test_kill_process_tree_does_not_raise_for_a_dead_pid():
    proc = subprocess.Popen(["/bin/bash", "-c", "exit 0"], start_new_session=True)
    proc.wait(timeout=5)

    kill_process_tree(proc.pid)
