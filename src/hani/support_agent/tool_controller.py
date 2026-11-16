"""Controls tool tabs: enable/disable, show/hide, reorder, move between panes.

Realises the operations that the ``Tool.move_*``/``close`` buttons (previously
``print()`` stubs) and the Support Agent both need. It operates on whichever
``pn.Tabs`` container is actually on screen:

* full view  — ``upper_tabs`` + ``lower_tabs`` + ``side_tabs`` are all displayed;
* simple view — only ``combined_tabs`` is displayed (``display_tabs``), and the
  three source tabs are off-screen, so we act directly on ``combined_tabs``.

Panes stored in ``Tabs.objects`` are the ``Tool`` instances themselves, so
enable/disable just flips ``tool.enabled`` (the same object is referenced by both
the source tab and the mirror, so it reflects in either view).

All methods assume they run on the IOLoop (the Support Agent marshals them via
``runtime.run_on_doc``); the human button handlers already run there.
"""

from __future__ import annotations

from typing import Any

import panel as pn

__all__ = ["ToolController"]

_PANE_KEYS = ("upper", "lower", "side")


class ToolController:
    def __init__(self, session_state: dict[str, Any]):
        self.session_state = session_state
        # Remembers hidden tabs so they can be restored:
        # name -> (name, pane, idx, container_key).
        self._hidden: dict[str, tuple[str, Any, int, str | None]] = {}
        # The enabled state of each hidden tool, so showing it again restores it
        # (removing a tab from the Tabs container can switch the tool off).
        self._was_enabled: dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    def _tabs(self, key: str) -> pn.Tabs | None:
        return self.session_state.get(f"{key}_tabs")

    def _display_containers(self) -> list[pn.Tabs]:
        """The Tabs widget(s) currently visible to the user."""
        ss = self.session_state
        display = ss.get("display_tabs")
        upper = ss.get("upper_tabs")
        if display is not None and display is not upper:
            # Simple view: a single combined Tabs is on screen.
            return [display]
        return [t for t in (upper, self._tabs("lower"), self._tabs("side")) if t is not None]

    @staticmethod
    def _names(tabs: pn.Tabs) -> list[str]:
        return list(getattr(tabs, "_names", None) or [])

    def _find(self, name: str):
        """Return (container, index, tab_name, tool) for a tool by (loose) name."""
        target = str(name).strip().lower()
        for container in self._display_containers():
            names = self._names(container)
            for i, (nm, pane) in enumerate(zip(names, container.objects)):
                if nm.lower() == target:
                    return container, i, nm, pane
        # Fall back to a contains-match for forgiving agent input.
        for container in self._display_containers():
            names = self._names(container)
            for i, (nm, pane) in enumerate(zip(names, container.objects)):
                if target in nm.lower():
                    return container, i, nm, pane
        return None

    # ------------------------------------------------------------------ #
    def list_tools(self) -> list[dict]:
        out: list[dict] = []
        for container in self._display_containers():
            pane_key = self._container_key(container)
            for nm, pane in zip(self._names(container), container.objects):
                out.append(
                    {
                        "name": nm,
                        "pane": pane_key,
                        "visible": True,
                        "enabled": bool(getattr(pane, "enabled", True)),
                        "permanent": bool(getattr(pane, "permanent", False)),
                    }
                )
        for nm, (_, pane, _, _) in self._hidden.items():
            out.append(
                {
                    "name": nm,
                    "pane": None,
                    "visible": False,
                    "enabled": bool(getattr(pane, "enabled", True)),
                    "permanent": bool(getattr(pane, "permanent", False)),
                }
            )
        return out

    def _container_key(self, container: pn.Tabs) -> str | None:
        for key in _PANE_KEYS:
            if self._tabs(key) is container:
                return key
        if container is self.session_state.get("display_tabs"):
            return "combined"
        return None

    # ------------------------------------------------------------------ #
    def set_tool_enabled(self, name: str, enabled: bool, allow_permanent: bool = False) -> dict:
        found = self._find(name)
        if found is None:
            return {"ok": False, "error": f"tool '{name}' not found"}
        _, _, nm, tool = found
        if getattr(tool, "permanent", False) and not allow_permanent:
            return {"ok": False, "error": f"'{nm}' is a permanent tool and cannot be disabled"}
        if not hasattr(tool, "enabled"):
            return {"ok": False, "error": f"'{nm}' cannot be toggled"}
        tool.enabled = bool(enabled)
        return {"ok": True, "name": nm, "enabled": bool(enabled)}

    def set_tool_visible(self, name: str, visible: bool, allow_permanent: bool = False) -> dict:
        """Show/hide a tool. ``allow_permanent`` lets the Support Agent (not the
        human close buttons) hide tools that are normally pinned; they stay
        restorable through ``set_tool_visible(name, True)`` / ``restore_all``."""
        if visible:
            return self._show(name)
        return self._hide(name, allow_permanent=allow_permanent)

    def restore_all(self) -> list[str]:
        """Bring back every tool the agent hid (e.g. when the user switches the
        agent off), so the participant is never left without their tools."""
        restored = []
        for nm in list(self._hidden):
            if self._show(nm).get("ok"):
                restored.append(nm)
        return restored

    def _hide(self, name: str, allow_permanent: bool = False) -> dict:
        found = self._find(name)
        if found is None:
            return {"ok": False, "error": f"tool '{name}' not found (already hidden?)"}
        container, idx, nm, pane = found
        if getattr(pane, "permanent", False) and not allow_permanent:
            return {"ok": False, "error": f"'{nm}' is permanent and cannot be hidden"}
        self._was_enabled[nm] = bool(getattr(pane, "enabled", True))
        container.pop(idx)
        self._hidden[nm] = (nm, pane, idx, self._container_key(container))
        return {"ok": True, "name": nm, "visible": False}

    def _container_by_key(self, key: str | None) -> pn.Tabs | None:
        if key in _PANE_KEYS:
            return self._tabs(key)
        if key == "combined":
            return self.session_state.get("display_tabs")
        containers = self._display_containers()
        return containers[0] if containers else None

    def _show(self, name: str) -> dict:
        match = None
        target_lc = str(name).strip().lower()
        for nm in self._hidden:
            if nm.lower() == target_lc or target_lc in nm.lower():
                match = nm
                break
        if match is None:
            return {"ok": True, "name": name, "visible": True, "note": "already visible"}
        nm, pane, idx, key = self._hidden.pop(match)
        # Restore into the original pane (falls back gracefully if unavailable).
        target = self._container_by_key(key)
        if target is None:
            return {"ok": False, "error": "no container to restore into"}
        pos = min(idx, len(target.objects))
        target.insert(pos, (nm, pane))
        if hasattr(pane, "enabled"):
            pane.enabled = self._was_enabled.pop(nm, True)
        return {"ok": True, "name": nm, "visible": True}

    # -- object-based entry points for the Tool move/close buttons ----------
    def _find_obj(self, tool):
        for container in self._display_containers():
            for i, (nm, pane) in enumerate(zip(self._names(container), container.objects)):
                if pane is tool:
                    return container, i, nm
        return None

    def move_obj_to_pane(self, tool, pane: str) -> dict:
        found = self._find_obj(tool)
        if found is None:
            return {"ok": False, "error": "tool not on screen"}
        return self.move_tool(found[2], position=0, pane=pane)

    def hide_obj(self, tool) -> dict:
        found = self._find_obj(tool)
        if found is None:
            return {"ok": False, "error": "tool not on screen"}
        return self.set_tool_visible(found[2], False)

    def move_tool(self, name: str, position: int | None = None, pane: str | None = None) -> dict:
        found = self._find(name)
        if found is None:
            return {"ok": False, "error": f"tool '{name}' not found"}
        container, idx, nm, tool = found
        containers = self._display_containers()
        single = len(containers) == 1  # simple/combined view

        # Resolve the destination container.
        dest = container
        if pane is not None and not single:
            dest_tabs = self._tabs(pane)
            if dest_tabs is None:
                return {"ok": False, "error": f"unknown pane '{pane}'"}
            dest = dest_tabs

        container.pop(idx)
        target = len(dest.objects) if position is None else max(0, min(int(position), len(dest.objects)))
        dest.insert(target, (nm, tool))
        return {
            "ok": True,
            "name": nm,
            "pane": self._container_key(dest) if not single else "combined",
            "position": target,
        }
