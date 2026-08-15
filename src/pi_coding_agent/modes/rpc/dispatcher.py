"""Command dispatch for the stdio RPC protocol.

Python port of `handleCommand()` and `handleInputLine()` in
`packages/coding-agent/src/modes/rpc/rpc-mode.ts`.

Dispatch lives apart from `rpc_mode.py` because everything interesting about
the protocol is here and none of it needs a process: `RpcDispatcher` takes an
`output` callable instead of owning stdout, so a test can drive all 34 commands
against a session and read the responses out of a list.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pi_tui.tasks import spawn

from ...core.extensions.types import UserBashEvent
from .types import (
    RPC_COMMAND_TYPES,
    RpcSessionState,
    RpcSlashCommand,
    make_error,
    make_success,
)
from .ui_context import RpcExtensionUIContext

if TYPE_CHECKING:
    from ...core.agent_session import AgentSession
    from ...core.agent_session_runtime import AgentSessionRuntime


class RpcDispatcher:
    """Turns decoded RPC commands into session calls and responses.

    `session` is read through `runtime_host` on every command rather than held,
    because `new_session`, `fork`, `clone` and `switch_session` all replace the
    session underneath us; a cached reference would keep driving the disposed
    one.
    """

    def __init__(
        self,
        runtime_host: AgentSessionRuntime,
        output: Callable[[dict[str, Any]], None],
        ui_context: RpcExtensionUIContext | None = None,
        rebind: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._runtime_host = runtime_host
        self._output = output
        self._ui_context = ui_context
        self._rebind = rebind
        self.shutdown_requested = False

    @property
    def session(self) -> AgentSession:
        return self._runtime_host.session

    def request_shutdown(self) -> None:
        self.shutdown_requested = True

    async def _after_replacement(self, result: dict[str, Any]) -> None:
        if not result.get("cancelled") and self._rebind is not None:
            await self._rebind()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    async def handle_input_line(self, line: str) -> None:
        """Decode one JSONL record and act on it.

        A malformed line answers with a `parse` error instead of terminating
        the loop -- a host that sends one bad line still has a usable session.
        """
        try:
            parsed = json.loads(line)
        except ValueError as parse_error:
            self._output(make_error(None, "parse", f"Failed to parse command: {parse_error}"))
            return

        if not isinstance(parsed, dict):
            self._output(make_error(None, "parse", "Command must be a JSON object"))
            return

        if parsed.get("type") == "extension_ui_response":
            if self._ui_context is not None:
                self._ui_context.resolve(parsed)
            return

        command_id = parsed.get("id")
        command_type = parsed.get("type")
        try:
            response = await self.handle_command(parsed)
        except Exception as command_error:
            message = str(command_error) or type(command_error).__name__
            self._output(make_error(command_id, str(command_type), message))
            return
        if response is not None:
            self._output(response)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def handle_command(self, command: dict[str, Any]) -> dict[str, Any] | None:
        command_id = command.get("id")
        command_type = command.get("type")
        if not isinstance(command_type, str) or command_type not in RPC_COMMAND_TYPES:
            return make_error(command_id, str(command_type), f"Unknown command: {command_type}")
        handler = getattr(self, f"_cmd_{command_type}")
        return await handler(command_id, command)

    # -- Prompting -----------------------------------------------------

    async def _cmd_prompt(self, command_id: str | None, command: dict[str, Any]) -> None:
        """Answer only once the prompt's preflight succeeds.

        The response cannot simply follow `prompt()` returning: that awaits the
        whole turn, and a host needs to know its prompt was accepted before the
        model has finished answering. So the success line is emitted from the
        preflight callback and the turn continues in the background; a failure
        before preflight succeeded becomes the error response instead.
        """
        preflight_succeeded = False

        def on_preflight(did_succeed: bool) -> None:
            nonlocal preflight_succeeded
            if did_succeed:
                preflight_succeeded = True
                self._output(make_success(command_id, "prompt"))

        session = self.session

        async def run() -> None:
            try:
                await session.prompt(
                    command.get("message", ""),
                    images=command.get("images"),
                    streaming_behavior=command.get("streamingBehavior"),
                    preflight_result=on_preflight,
                    source="rpc",
                )
            except Exception as prompt_error:
                if not preflight_succeeded:
                    message = str(prompt_error) or type(prompt_error).__name__
                    self._output(make_error(command_id, "prompt", message))

        spawn(run())
        return None

    async def _cmd_steer(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        await self.session.steer(command.get("message", ""), command.get("images"))
        return make_success(command_id, "steer")

    async def _cmd_follow_up(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        await self.session.follow_up(command.get("message", ""), command.get("images"))
        return make_success(command_id, "follow_up")

    async def _cmd_abort(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        await self.session.abort()
        return make_success(command_id, "abort")

    async def _cmd_new_session(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        result = await self._runtime_host.new_session(command.get("parentSession"))
        await self._after_replacement(result)
        return make_success(command_id, "new_session", {"cancelled": bool(result.get("cancelled"))})

    # -- State ---------------------------------------------------------

    async def _cmd_get_state(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        session = self.session
        state = RpcSessionState(
            model=session.model,
            thinking_level=session.thinking_level,
            is_streaming=session.is_streaming,
            is_compacting=session.is_compacting,
            steering_mode=session.steering_mode,
            follow_up_mode=session.follow_up_mode,
            session_file=session.session_file,
            session_id=session.session_id,
            session_name=session.session_name,
            auto_compaction_enabled=session.auto_compaction_enabled,
            message_count=len(session.messages),
            pending_message_count=session.pending_message_count,
        )
        return make_success(command_id, "get_state", state)

    # -- Model ---------------------------------------------------------

    async def _cmd_set_model(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        provider = command.get("provider")
        model_id = command.get("modelId")
        session = self.session
        for model in session.model_runtime.get_available_snapshot():
            if model.provider == provider and model.id == model_id:
                await session.set_model(model)
                return make_success(command_id, "set_model", model)
        return make_error(command_id, "set_model", f"Model not found: {provider}/{model_id}")

    async def _cmd_cycle_model(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        result = await self.session.cycle_model()
        return make_success(command_id, "cycle_model", result if result is not None else None)

    async def _cmd_get_available_models(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        models = self.session.model_runtime.get_available_snapshot()
        return make_success(command_id, "get_available_models", {"models": models})

    # -- Thinking ------------------------------------------------------

    async def _cmd_set_thinking_level(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        self.session.set_thinking_level(command.get("level"))
        return make_success(command_id, "set_thinking_level")

    async def _cmd_cycle_thinking_level(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        level = self.session.cycle_thinking_level()
        if level is None:
            return make_success(command_id, "cycle_thinking_level", None)
        return make_success(command_id, "cycle_thinking_level", {"level": level})

    async def _cmd_get_available_thinking_levels(
        self, command_id: str | None, _command: dict[str, Any]
    ) -> dict[str, Any]:
        levels = self.session.get_available_thinking_levels()
        return make_success(command_id, "get_available_thinking_levels", {"levels": list(levels)})

    # -- Queue modes ---------------------------------------------------

    async def _cmd_set_steering_mode(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        self.session.set_steering_mode(command.get("mode"))
        return make_success(command_id, "set_steering_mode")

    async def _cmd_set_follow_up_mode(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        self.session.set_follow_up_mode(command.get("mode"))
        return make_success(command_id, "set_follow_up_mode")

    # -- Compaction ----------------------------------------------------

    async def _cmd_compact(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        result = await self.session.compact(command.get("customInstructions"))
        return make_success(command_id, "compact", result)

    async def _cmd_set_auto_compaction(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        self.session.set_auto_compaction_enabled(bool(command.get("enabled")))
        return make_success(command_id, "set_auto_compaction")

    # -- Retry ---------------------------------------------------------

    async def _cmd_set_auto_retry(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        self.session.set_auto_retry_enabled(bool(command.get("enabled")))
        return make_success(command_id, "set_auto_retry")

    async def _cmd_abort_retry(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        self.session.abort_retry()
        return make_success(command_id, "abort_retry")

    # -- Bash ----------------------------------------------------------

    async def _cmd_bash(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        session = self.session
        shell_command = command.get("command", "")
        exclude_from_context = bool(command.get("excludeFromContext", False))

        event_result = await session.extension_runner.emit_user_bash(
            UserBashEvent(
                command=shell_command,
                exclude_from_context=exclude_from_context,
                cwd=session.session_manager.get_cwd(),
            )
        )

        # A handler that returns a complete result has already run the command
        # somewhere else; recording it keeps it in the transcript without
        # running it a second time here.
        if event_result is not None and event_result.result is not None:
            session.record_bash_result(shell_command, event_result.result, exclude_from_context=exclude_from_context)
            return make_success(command_id, "bash", event_result.result)

        result = await session.execute_bash(
            shell_command,
            exclude_from_context=exclude_from_context,
            id=command_id,
            operations=event_result.operations if event_result is not None else None,
        )
        return make_success(command_id, "bash", result)

    async def _cmd_abort_bash(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        self.session.abort_bash()
        return make_success(command_id, "abort_bash")

    # -- Session -------------------------------------------------------

    async def _cmd_get_session_stats(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        return make_success(command_id, "get_session_stats", self.session.get_session_stats())

    async def _cmd_export_html(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        path = await self.session.export_to_html(command.get("outputPath"))
        return make_success(command_id, "export_html", {"path": path})

    async def _cmd_switch_session(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        result = await self._runtime_host.switch_session(command.get("sessionPath", ""))
        await self._after_replacement(result)
        return make_success(command_id, "switch_session", {"cancelled": bool(result.get("cancelled"))})

    async def _cmd_fork(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        result = await self._runtime_host.fork(command.get("entryId", ""))
        await self._after_replacement(result)
        return make_success(
            command_id,
            "fork",
            {"text": result.get("selected_text"), "cancelled": bool(result.get("cancelled"))},
        )

    async def _cmd_clone(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        leaf_id = self.session.session_manager.get_leaf_id()
        if not leaf_id:
            return make_error(command_id, "clone", "Cannot clone session: no current entry selected")
        result = await self._runtime_host.fork(leaf_id, position="at")
        await self._after_replacement(result)
        return make_success(command_id, "clone", {"cancelled": bool(result.get("cancelled"))})

    async def _cmd_get_fork_messages(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        messages = self.session.get_user_messages_for_forking()
        return make_success(command_id, "get_fork_messages", {"messages": messages})

    async def _cmd_get_entries(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        session_manager = self.session.session_manager
        entries = session_manager.get_entries()
        since = command.get("since")
        if since is not None:
            for index, entry in enumerate(entries):
                if entry.id == since:
                    entries = entries[index + 1 :]
                    break
            else:
                return make_error(command_id, "get_entries", f"Entry not found: {since}")
        return make_success(
            command_id,
            "get_entries",
            {"entries": entries, "leafId": session_manager.get_leaf_id()},
        )

    async def _cmd_get_tree(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        session_manager = self.session.session_manager
        return make_success(
            command_id,
            "get_tree",
            {"tree": session_manager.get_tree(), "leafId": session_manager.get_leaf_id()},
        )

    async def _cmd_get_last_assistant_text(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        return make_success(command_id, "get_last_assistant_text", {"text": self.session.get_last_assistant_text()})

    async def _cmd_set_session_name(self, command_id: str | None, command: dict[str, Any]) -> dict[str, Any]:
        name = str(command.get("name", "")).strip()
        if not name:
            return make_error(command_id, "set_session_name", "Session name cannot be empty")
        self.session.set_session_name(name)
        return make_success(command_id, "set_session_name")

    # -- Messages ------------------------------------------------------

    async def _cmd_get_messages(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        return make_success(command_id, "get_messages", {"messages": self.session.messages})

    # -- Commands ------------------------------------------------------

    async def _cmd_get_commands(self, command_id: str | None, _command: dict[str, Any]) -> dict[str, Any]:
        session = self.session
        commands: list[RpcSlashCommand] = []

        for invocation_name, registered in session.extension_runner.get_registered_commands():
            commands.append(
                RpcSlashCommand(
                    name=invocation_name,
                    description=registered.description,
                    source="extension",
                    source_info=registered.source_info,
                )
            )

        for template in session.prompt_templates:
            commands.append(
                RpcSlashCommand(
                    name=template.name,
                    description=template.description,
                    source="prompt",
                    source_info=template.source_info,
                )
            )

        with contextlib.suppress(Exception):
            for skill in session.resource_loader.get_skills().skills:
                commands.append(
                    RpcSlashCommand(
                        name=f"skill:{skill.name}",
                        description=skill.description,
                        source="skill",
                        source_info=skill.source_info,
                    )
                )

        return make_success(command_id, "get_commands", {"commands": commands})


__all__ = ["RpcDispatcher"]
