"""The `/config` dialog: enable/disable package resources.

Ported from ``packages/coding-agent/src/modes/interactive/components/config-selector.ts``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pi_tui.component import Component, Container
from pi_tui.components.input import Input
from pi_tui.components.spacer import Spacer
from pi_tui.keybindings import get_keybindings
from pi_tui.keys import matches_key
from pi_tui.utils import truncate_to_width, visible_width

from ....core.config import CONFIG_DIR_NAME
from ....utils.paths import canonicalize_path, is_local_path, resolve_path
from ..theme.theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, raw_key_hint

if TYPE_CHECKING:
    from ....core.settings_manager import SettingsManager

ResourceType = Literal["extensions", "skills", "prompts", "themes"]
ConfigWriteScope = Literal["global", "project"]
SettingsScope = Literal["user", "project"]
ProjectOverrideState = Literal["inherit", "load", "unload"]

RESOURCE_TYPES: tuple[ResourceType, ...] = ("extensions", "skills", "prompts", "themes")

RESOURCE_TYPE_LABELS: dict[str, str] = {
    "extensions": "Extensions",
    "skills": "Skills",
    "prompts": "Prompts",
    "themes": "Themes",
}

_TYPE_ORDER: dict[str, int] = {"extensions": 0, "skills": 1, "prompts": 2, "themes": 3}


def format_base_dir(base_dir: str) -> str:
    home_dir = str(Path.home())
    if base_dir == home_dir:
        display_path = "~"
    elif base_dir.startswith(home_dir):
        display_path = f"~{base_dir[len(home_dir) :].replace(chr(92), '/')}"
    else:
        display_path = base_dir.replace("\\", "/")
    return display_path if display_path.endswith("/") else f"{display_path}/"


def get_group_label(metadata: Any, agent_dir: str) -> str:
    if metadata.origin == "package":
        return f"{metadata.source} ({metadata.scope})"
    if metadata.source == "auto":
        if metadata.base_dir:
            return (
                f"User ({format_base_dir(metadata.base_dir)})"
                if metadata.scope == "user"
                else f"Project ({format_base_dir(metadata.base_dir)})"
            )
        return f"User ({format_base_dir(agent_dir)})" if metadata.scope == "user" else f"Project ({CONFIG_DIR_NAME}/)"
    return "User settings" if metadata.scope == "user" else "Project settings"


@dataclass
class ResourceItem:
    path: str
    enabled: bool
    metadata: Any
    resource_type: ResourceType
    display_name: str
    group_key: str
    subgroup_key: str


@dataclass
class ResourceSubgroup:
    type: ResourceType
    label: str
    items: list[ResourceItem] = field(default_factory=list)


@dataclass
class ResourceGroup:
    key: str
    label: str
    scope: str
    origin: str
    source: str
    subgroups: list[ResourceSubgroup] = field(default_factory=list)


@dataclass
class FlatEntry:
    type: Literal["group", "subgroup", "item"]
    group: ResourceGroup | None = None
    subgroup: ResourceSubgroup | None = None
    item: ResourceItem | None = None


def build_groups(resolved: Any, agent_dir: str) -> list[ResourceGroup]:
    group_map: dict[str, ResourceGroup] = {}

    def add_to_group(resources: list[Any], resource_type: ResourceType) -> None:
        for resource in resources:
            metadata = resource.metadata
            base_dir = getattr(metadata, "base_dir", None) or ""
            group_key = f"{metadata.origin}:{metadata.scope}:{metadata.source}:{base_dir}"

            if group_key not in group_map:
                group_map[group_key] = ResourceGroup(
                    key=group_key,
                    label=get_group_label(metadata, agent_dir),
                    scope=metadata.scope,
                    origin=metadata.origin,
                    source=metadata.source,
                )
            group = group_map[group_key]

            subgroup = next((sg for sg in group.subgroups if sg.type == resource_type), None)
            if subgroup is None:
                subgroup = ResourceSubgroup(type=resource_type, label=RESOURCE_TYPE_LABELS[resource_type])
                group.subgroups.append(subgroup)

            file_name = os.path.basename(resource.path)
            parent_folder = os.path.basename(os.path.dirname(resource.path))
            if resource_type == "extensions" and parent_folder != "extensions":
                display_name = f"{parent_folder}/{file_name}"
            elif resource_type == "skills" and file_name == "SKILL.md":
                display_name = parent_folder
            else:
                display_name = file_name

            subgroup.items.append(
                ResourceItem(
                    path=resource.path,
                    enabled=resource.enabled,
                    metadata=metadata,
                    resource_type=resource_type,
                    display_name=display_name,
                    group_key=group_key,
                    subgroup_key=f"{group_key}:{resource_type}",
                )
            )

    for resource_type in RESOURCE_TYPES:
        add_to_group(getattr(resolved, resource_type, []) or [], resource_type)

    groups = list(group_map.values())
    # Packages first, then top-level; user before project; then by source.
    groups.sort(
        key=lambda group: (
            0 if group.origin == "package" else 1,
            0 if group.scope == "user" else 1,
            group.source,
        )
    )
    for group in groups:
        group.subgroups.sort(key=lambda subgroup: _TYPE_ORDER[subgroup.type])
        for subgroup in group.subgroups:
            subgroup.items.sort(key=lambda item: item.display_name)
    return groups


class ConfigSelectorHeader(Component):
    def __init__(self, write_scope: ConfigWriteScope, project_mode_available: bool) -> None:
        self.write_scope = write_scope
        self.project_mode_available = project_mode_available

    def set_write_scope(self, write_scope: ConfigWriteScope) -> None:
        self.write_scope = write_scope

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        title = theme.bold("Project Local Resources" if self.write_scope == "project" else "Global Resources")
        separator = theme.fg("muted", " · ")
        switch_hint = key_hint("tui.input.tab", "switch mode") + separator if self.project_mode_available else ""
        action_hint = (
            raw_key_hint("space", "cycle inherit/+/-")
            if self.write_scope == "project"
            else raw_key_hint("space", "toggle")
        )
        hint = switch_hint + action_hint + separator + raw_key_hint("esc", "close")
        spacing = max(1, width - visible_width(title) - visible_width(hint))
        scope_hint = theme.fg(
            "muted",
            f"{CONFIG_DIR_NAME}/settings.json · inherited global resources are dimmed"
            if self.write_scope == "project"
            else f"~/{CONFIG_DIR_NAME}/agent/settings.json",
        )
        return [
            truncate_to_width(f"{title}{' ' * spacing}{hint}", width, ""),
            truncate_to_width(scope_hint, width, ""),
        ]


class ResourceList(Component):
    def __init__(
        self,
        groups_by_scope: dict[str, list[ResourceGroup]],
        settings_manager: SettingsManager,
        cwd: str,
        agent_dir: str,
        terminal_height: int | None = None,
        write_scope: ConfigWriteScope = "global",
    ) -> None:
        self.groups_by_scope = groups_by_scope
        self.settings_manager = settings_manager
        self.cwd = cwd
        self.agent_dir = agent_dir
        self.write_scope = write_scope
        self.inherited_enabled_by_key = self._build_inherited_enabled_map(groups_by_scope["global"])
        self.search_input = Input()
        self.flat_items: list[FlatEntry] = []
        self.filtered_items: list[FlatEntry] = []
        self.selected_index = 0
        self._focused = False

        self.on_cancel: Callable[[], None] | None = None
        self.on_exit: Callable[[], None] | None = None
        self.on_toggle: Callable[[ResourceItem, bool], None] | None = None
        self.on_switch_mode: Callable[[], None] | None = None

        # 8 lines of chrome: spacer + border + spacer + 2-line header + spacer
        # + bottom spacer + bottom border.
        self.max_visible = max(5, (terminal_height or 24) - 8)
        self._build_flat_list()
        self.filtered_items = list(self.flat_items)

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.search_input.focused = value

    @property
    def groups(self) -> list[ResourceGroup]:
        return self.groups_by_scope[self.write_scope]

    def set_write_scope(self, write_scope: ConfigWriteScope) -> None:
        self.write_scope = write_scope
        self._build_flat_list()
        self._filter_items(self.search_input.get_value())

    def _build_inherited_enabled_map(self, groups: list[ResourceGroup]) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for group in groups:
            for subgroup in group.subgroups:
                for item in subgroup.items:
                    result[self._get_resource_item_key(item)] = item.enabled
        return result

    def _build_flat_list(self) -> None:
        self.flat_items = []
        for group in self.groups:
            self.flat_items.append(FlatEntry(type="group", group=group))
            for subgroup in group.subgroups:
                self.flat_items.append(FlatEntry(type="subgroup", subgroup=subgroup, group=group))
                for item in subgroup.items:
                    self.flat_items.append(FlatEntry(type="item", item=item))
        self.selected_index = next((i for i, entry in enumerate(self.flat_items) if entry.type == "item"), 0)

    def _find_next_item(self, from_index: int, direction: int) -> int:
        index = from_index + direction
        while 0 <= index < len(self.filtered_items):
            if self.filtered_items[index].type == "item":
                return index
            index += direction
        return from_index

    def _filter_items(self, query: str) -> None:
        if not query.strip():
            self.filtered_items = list(self.flat_items)
            self._select_first_item()
            return

        lower_query = query.lower()
        matching_items: set[int] = set()
        for entry in self.flat_items:
            if entry.type == "item" and entry.item is not None:
                item = entry.item
                if (
                    lower_query in item.display_name.lower()
                    or lower_query in item.resource_type.lower()
                    or lower_query in item.path.lower()
                ):
                    matching_items.add(id(item))

        matching_subgroups: set[int] = set()
        matching_groups: set[int] = set()
        for group in self.groups:
            for subgroup in group.subgroups:
                for item in subgroup.items:
                    if id(item) in matching_items:
                        matching_subgroups.add(id(subgroup))
                        matching_groups.add(id(group))

        keep_by_type = {
            "group": lambda entry: id(entry.group) in matching_groups,
            "subgroup": lambda entry: id(entry.subgroup) in matching_subgroups,
            "item": lambda entry: id(entry.item) in matching_items,
        }
        self.filtered_items = [entry for entry in self.flat_items if keep_by_type[entry.type](entry)]

        self._select_first_item()

    def _select_first_item(self) -> None:
        self.selected_index = next((i for i, entry in enumerate(self.filtered_items) if entry.type == "item"), 0)

    def update_item(self, item: ResourceItem, enabled: bool) -> None:
        item.enabled = enabled
        for group in self.groups:
            for subgroup in group.subgroups:
                for candidate in subgroup.items:
                    if candidate.path == item.path and candidate.resource_type == item.resource_type:
                        candidate.enabled = enabled
                        return

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        lines.extend(self.search_input.render(width))
        lines.append("")

        if len(self.filtered_items) == 0:
            lines.append(theme.fg("muted", "  No resources found"))
            return lines

        start_index = max(
            0,
            min(
                self.selected_index - math.floor(self.max_visible / 2),
                len(self.filtered_items) - self.max_visible,
            ),
        )
        end_index = min(start_index + self.max_visible, len(self.filtered_items))

        for index in range(start_index, end_index):
            entry = self.filtered_items[index]
            is_selected = index == self.selected_index

            if entry.type == "group" and entry.group is not None:
                inherited = self.write_scope == "project" and entry.group.scope == "user"
                label = theme.bold(f"{entry.group.label}{' · inherited global' if inherited else ''}")
                lines.append(truncate_to_width(f"  {theme.fg('dim' if inherited else 'accent', label)}", width, ""))
            elif entry.type == "subgroup" and entry.subgroup is not None:
                color = (
                    "dim"
                    if self.write_scope == "project" and entry.group is not None and entry.group.scope == "user"
                    else "muted"
                )
                lines.append(truncate_to_width(f"    {theme.fg(color, entry.subgroup.label)}", width, ""))
            elif entry.item is not None:
                item = entry.item
                cursor = "> " if is_selected else "  "
                dimmed = self._is_dimmed_item(item)
                name_text = theme.bold(item.display_name) if is_selected and not dimmed else item.display_name
                name = theme.fg("dim", name_text) if dimmed else name_text
                lines.append(
                    truncate_to_width(
                        f"{cursor}    {self._render_checkbox(item)} {name}{self._get_item_suffix(item)}",
                        width,
                        "...",
                    )
                )

        if start_index > 0 or end_index < len(self.filtered_items):
            item_count = sum(1 for entry in self.filtered_items if entry.type == "item")
            current_item_index = (
                sum(1 for entry in self.filtered_items[: self.selected_index] if entry.type == "item") + 1
            )
            lines.append(theme.fg("dim", f"  ({current_item_index}/{item_count})"))

        return lines

    def handle_input(self, data: str) -> None:
        keybindings = get_keybindings()

        if keybindings.matches(data, "tui.select.up"):
            self.selected_index = self._find_next_item(self.selected_index, -1)
            return
        if keybindings.matches(data, "tui.select.down"):
            self.selected_index = self._find_next_item(self.selected_index, 1)
            return
        if keybindings.matches(data, "tui.select.pageUp"):
            target = max(0, self.selected_index - self.max_visible)
            while target < len(self.filtered_items) and self.filtered_items[target].type != "item":
                target += 1
            if target < len(self.filtered_items):
                self.selected_index = target
            return
        if keybindings.matches(data, "tui.select.pageDown"):
            target = min(len(self.filtered_items) - 1, self.selected_index + self.max_visible)
            while target >= 0 and self.filtered_items[target].type != "item":
                target -= 1
            if target >= 0:
                self.selected_index = target
            return
        if keybindings.matches(data, "tui.select.cancel"):
            if self.on_cancel is not None:
                self.on_cancel()
            return
        if matches_key(data, "ctrl+c"):
            if self.on_exit is not None:
                self.on_exit()
            return
        if keybindings.matches(data, "tui.input.tab"):
            if self.on_switch_mode is not None:
                self.on_switch_mode()
            return
        if data == " " or keybindings.matches(data, "tui.select.confirm"):
            if 0 <= self.selected_index < len(self.filtered_items):
                entry = self.filtered_items[self.selected_index]
                if entry.type == "item" and entry.item is not None:
                    item = entry.item
                    if self.write_scope == "project" or self._get_item_scope(item) == "user":
                        new_enabled = self._toggle_resource(item)
                        if new_enabled is not None:
                            self.update_item(item, new_enabled)
                            if self.on_toggle is not None:
                                self.on_toggle(item, new_enabled)
            return

        self.search_input.handle_input(data)
        self._filter_items(self.search_input.get_value())

    # -- toggling -----------------------------------------------------------

    def _toggle_resource(self, item: ResourceItem) -> bool | None:
        if self.write_scope == "project":
            state = self._get_next_override_state(item)
            if not self._set_project_resource_override(item, state):
                return None
            return self._get_inherited_enabled(item) if state == "inherit" else state == "load"

        enabled = not item.enabled
        if item.metadata.origin == "top-level":
            self._toggle_top_level_resource(item, enabled)
        else:
            self._toggle_package_resource(item, enabled)
        return enabled

    def _filter_out_pattern(self, entries: list[str], pattern: str) -> list[str]:
        return [entry for entry in entries if self._get_pattern_entry_target(entry) != pattern]

    def _toggle_top_level_resource(self, item: ResourceItem, enabled: bool) -> None:
        scope = "project" if item.metadata.scope == "project" else "user"
        settings = (
            self.settings_manager.get_project_settings()
            if scope == "project"
            else self.settings_manager.get_global_settings()
        )
        current = list(settings.get(item.resource_type) or [])
        pattern = self._get_resource_pattern(item)
        updated = self._filter_out_pattern(current, pattern)
        updated.append(f"{'+' if enabled else '-'}{pattern}")
        self._set_paths(item.resource_type, updated, scope)

    def _set_paths(self, resource_type: ResourceType, paths: list[str], scope: SettingsScope) -> None:
        prefix = "set_project_" if scope == "project" else "set_"
        setter_names = {
            "extensions": f"{prefix}extension_paths",
            "skills": f"{prefix}skill_paths",
            "prompts": f"{prefix}prompt_template_paths",
            "themes": f"{prefix}theme_paths",
        }
        getattr(self.settings_manager, setter_names[resource_type])(paths)

    def _toggle_package_resource(self, item: ResourceItem, enabled: bool) -> None:
        scope = "project" if item.metadata.scope == "project" else "user"
        settings = (
            self.settings_manager.get_project_settings()
            if scope == "project"
            else self.settings_manager.get_global_settings()
        )
        packages = list(settings.get("packages") or [])
        package_index = next(
            (
                index
                for index, package in enumerate(packages)
                if (package if isinstance(package, str) else package.get("source")) == item.metadata.source
            ),
            -1,
        )
        if package_index == -1:
            return

        package = packages[package_index]
        if isinstance(package, str):
            package = {"source": package}
            packages[package_index] = package

        pattern = self._get_package_resource_pattern(item)
        updated = self._filter_out_pattern(list(package.get(item.resource_type) or []), pattern)
        updated.append(f"{'+' if enabled else '-'}{pattern}")
        package[item.resource_type] = updated if len(updated) > 0 else None

        if not any(package.get(key) is not None for key in RESOURCE_TYPES):
            packages[package_index] = package["source"]

        if scope == "project":
            self.settings_manager.set_project_packages(packages)
        else:
            self.settings_manager.set_packages(packages)

    # -- rendering helpers --------------------------------------------------

    def _render_checkbox(self, item: ResourceItem) -> str:
        if self.write_scope == "project":
            state = self._get_project_override_state(item)
            if state == "load":
                return theme.fg("success", "[+]")
            if state == "unload":
                return theme.fg("warning", "[-]")
            return theme.fg("dim", "[x]" if item.enabled else "[ ]")
        return theme.fg("success", "[x]") if item.enabled else theme.fg("dim", "[ ]")

    def _get_item_suffix(self, item: ResourceItem) -> str:
        if self.write_scope != "project":
            return ""
        state = self._get_project_override_state(item)
        if state == "load":
            return theme.fg("muted", "  project load")
        if state == "unload":
            return theme.fg("muted", "  project unload")
        return theme.fg("dim", "  inherited global") if self._is_inherited_global_item(item) else ""

    def _is_dimmed_item(self, item: ResourceItem) -> bool:
        return (
            self.write_scope == "project"
            and self._is_inherited_global_item(item)
            and self._get_project_override_state(item) == "inherit"
        )

    # -- project overrides --------------------------------------------------

    def _set_project_resource_override(self, item: ResourceItem, state: ProjectOverrideState) -> bool:
        if item.metadata.origin == "top-level":
            return self._set_project_top_level_override(item, state)
        return self._set_project_package_override(item, state)

    def _set_project_top_level_override(self, item: ResourceItem, state: ProjectOverrideState) -> bool:
        current = list(self.settings_manager.get_project_settings().get(item.resource_type) or [])
        pattern = (
            item.path if self._is_inherited_global_item(item) else self._get_resource_pattern_for_scope(item, "project")
        )
        patterns = self._get_top_level_override_patterns(item, "project")

        updated: list[str] = []
        for entry in current:
            target = self._get_pattern_entry_target(entry)
            if entry.startswith(("!", "+", "-")) and target in patterns:
                continue
            if state == "inherit" and self._is_inherited_global_item(item) and target == pattern:
                continue
            updated.append(entry)

        if state != "inherit":
            if self._is_inherited_global_item(item) and pattern not in updated:
                updated.append(pattern)
            updated.append(f"{'+' if state == 'load' else '-'}{pattern}")

        self._set_paths(item.resource_type, updated, "project")
        return True

    def _set_project_package_override(self, item: ResourceItem, state: ProjectOverrideState) -> bool:
        packages = list(self.settings_manager.get_project_settings().get("packages") or [])
        package_index = next(
            (
                index
                for index, package in enumerate(packages)
                if self._package_source_string_matches(
                    item.metadata.source,
                    self._get_item_scope(item),
                    package if isinstance(package, str) else package.get("source", ""),
                    "project",
                )
            ),
            -1,
        )
        if package_index == -1:
            if state == "inherit":
                return False
            packages.append(self._create_package_override_source(item))
            package_index = len(packages) - 1

        package = packages[package_index]
        if isinstance(package, str):
            package = {"source": package}
            packages[package_index] = package

        pattern = self._get_package_resource_pattern(item)
        updated = self._filter_out_pattern(list(package.get(item.resource_type) or []), pattern)
        if state != "inherit":
            updated.append(f"{'+' if state == 'load' else '-'}{pattern}")
        package[item.resource_type] = updated if len(updated) > 0 else None

        if not any(package.get(key) is not None for key in RESOURCE_TYPES):
            if package.get("autoload") is False:
                packages.pop(package_index)
            else:
                packages[package_index] = package["source"]

        self.settings_manager.set_project_packages(packages)
        return True

    def _get_next_override_state(self, item: ResourceItem) -> ProjectOverrideState:
        state = self._get_project_override_state(item)
        inherited_enabled = self._get_inherited_enabled(item)
        if state == "inherit":
            return "unload" if inherited_enabled else "load"
        if state == "unload":
            return "load" if inherited_enabled else "inherit"
        return "inherit" if inherited_enabled else "unload"

    def _get_project_override_state(self, item: ResourceItem) -> ProjectOverrideState:
        if self.write_scope != "project":
            return "inherit"
        if item.metadata.origin == "top-level":
            return self._get_override_state_from_entries(
                list(self.settings_manager.get_project_settings().get(item.resource_type) or []),
                self._get_top_level_override_patterns(item, "project"),
                False,
            )
        package = self._find_matching_package_source(item, "project")
        if not isinstance(package, dict):
            return "inherit"
        entries = package.get(item.resource_type)
        if entries is None:
            return "inherit"
        return self._get_override_state_from_entries(
            entries, {self._get_package_resource_pattern(item)}, package.get("autoload") is not False
        )

    def _get_override_state_from_entries(
        self, entries: list[str], patterns: set[str], empty_array_is_unload: bool
    ) -> ProjectOverrideState:
        if len(entries) == 0 and empty_array_is_unload:
            return "unload"
        state: ProjectOverrideState = "inherit"
        for entry in entries:
            if self._get_pattern_entry_target(entry) not in patterns:
                continue
            state = "unload" if entry.startswith(("!", "-")) else "load"
        return state

    def _get_inherited_enabled(self, item: ResourceItem) -> bool:
        key = self._get_resource_item_key(item)
        if key in self.inherited_enabled_by_key:
            return self.inherited_enabled_by_key[key]
        return item.enabled if self._get_item_scope(item) == "user" else True

    def _is_inherited_global_item(self, item: ResourceItem) -> bool:
        return (
            self._get_item_scope(item) == "user" or self._get_resource_item_key(item) in self.inherited_enabled_by_key
        )

    def _get_top_level_override_patterns(self, item: ResourceItem, scope: SettingsScope) -> set[str]:
        base_dir = self._get_top_level_base_dir(scope)
        patterns = {
            self._get_resource_pattern_for_scope(item, scope),
            item.path,
            os.path.relpath(item.path, base_dir),
        }
        item_base_dir = getattr(item.metadata, "base_dir", None)
        if item_base_dir:
            patterns.add(os.path.relpath(item.path, item_base_dir))
        return patterns

    def _get_resource_pattern_for_scope(self, item: ResourceItem, scope: SettingsScope) -> str:
        source_scope = self._get_item_scope(item)
        if scope != source_scope:
            return item.path
        base_dir = getattr(item.metadata, "base_dir", None) or self._get_top_level_base_dir(source_scope)
        return os.path.relpath(item.path, base_dir)

    def _create_package_override_source(self, item: ResourceItem) -> dict[str, Any]:
        source = item.metadata.source
        if not is_local_path(source):
            return {"source": source, "autoload": False}
        source_path = resolve_path(source, self._get_top_level_base_dir(self._get_item_scope(item)))
        relative = os.path.relpath(source_path, self._get_top_level_base_dir("project"))
        return {"source": relative or ".", "autoload": False}

    def _package_source_string_matches(
        self, left_source: str, left_scope: SettingsScope, right_source: str, right_scope: SettingsScope
    ) -> bool:
        if left_source == right_source:
            return True
        if not is_local_path(left_source) or not is_local_path(right_source):
            return False
        left = resolve_path(left_source, self._get_top_level_base_dir(left_scope))
        right = resolve_path(right_source, self._get_top_level_base_dir(right_scope))
        return left == right

    def _find_matching_package_source(self, item: ResourceItem, target_scope: SettingsScope) -> Any:
        settings = (
            self.settings_manager.get_project_settings()
            if target_scope == "project"
            else self.settings_manager.get_global_settings()
        )
        for package in settings.get("packages") or []:
            source = package if isinstance(package, str) else package.get("source", "")
            if self._package_source_string_matches(
                item.metadata.source, self._get_item_scope(item), source, target_scope
            ):
                return package
        return None

    def _get_pattern_entry_target(self, entry: str) -> str:
        return entry[1:] if entry.startswith(("!", "+", "-")) else entry

    def _get_resource_item_key(self, item: ResourceItem) -> str:
        return f"{item.resource_type}:{canonicalize_path(item.path)}"

    def _get_item_scope(self, item: ResourceItem) -> SettingsScope:
        return "project" if item.metadata.scope == "project" else "user"

    def _get_top_level_base_dir(self, scope: SettingsScope) -> str:
        return os.path.join(self.cwd, CONFIG_DIR_NAME) if scope == "project" else self.agent_dir

    def _get_resource_pattern(self, item: ResourceItem) -> str:
        scope = "project" if item.metadata.scope == "project" else "user"
        base_dir = getattr(item.metadata, "base_dir", None) or self._get_top_level_base_dir(scope)
        return os.path.relpath(item.path, base_dir)

    def _get_package_resource_pattern(self, item: ResourceItem) -> str:
        base_dir = getattr(item.metadata, "base_dir", None) or os.path.dirname(item.path)
        return os.path.relpath(item.path, base_dir)


class ConfigSelectorComponent(Container):
    def __init__(
        self,
        resolved_paths: dict[str, Any],
        settings_manager: SettingsManager,
        cwd: str,
        agent_dir: str,
        on_close: Callable[[], None],
        on_exit: Callable[[], None],
        request_render: Callable[[], None],
        terminal_height: int | None = None,
        write_scope: ConfigWriteScope = "global",
        project_mode_available: bool = True,
    ) -> None:
        super().__init__()
        self.write_scope = write_scope
        self._focused = False

        groups_by_scope = {
            "global": build_groups(resolved_paths["global"], agent_dir),
            "project": build_groups(resolved_paths["project"], agent_dir),
        }

        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.header = ConfigSelectorHeader(self.write_scope, project_mode_available)
        self.add_child(self.header)
        self.add_child(Spacer(1))

        self.resource_list = ResourceList(
            groups_by_scope, settings_manager, cwd, agent_dir, terminal_height, self.write_scope
        )
        self.resource_list.on_cancel = on_close
        self.resource_list.on_exit = on_exit
        self.resource_list.on_toggle = lambda _item, _enabled: request_render()
        if project_mode_available:

            def switch() -> None:
                self._switch_write_scope()
                request_render()

            self.resource_list.on_switch_mode = switch
        self.add_child(self.resource_list)

        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.resource_list.focused = value

    def _switch_write_scope(self) -> None:
        self.write_scope = "project" if self.write_scope == "global" else "global"
        self.header.set_write_scope(self.write_scope)
        self.resource_list.set_write_scope(self.write_scope)

    def get_resource_list(self) -> ResourceList:
        return self.resource_list

    def handle_input(self, data: str) -> None:
        self.resource_list.handle_input(data)


__all__ = [
    "RESOURCE_TYPES",
    "RESOURCE_TYPE_LABELS",
    "ConfigSelectorComponent",
    "ConfigSelectorHeader",
    "FlatEntry",
    "ResourceGroup",
    "ResourceItem",
    "ResourceList",
    "ResourceSubgroup",
    "build_groups",
    "format_base_dir",
    "get_group_label",
]
