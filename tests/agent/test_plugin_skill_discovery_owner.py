"""Regression coverage for canonical plugin skill discovery ownership."""

import builtins
from unittest.mock import patch

from agent.skill_commands import get_plugin_skill_commands, invalidate_plugin_skill_commands
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_plugin_skill_commands_uses_runtime_discovery_owner(tmp_path):
    """A live invocation must not import retired hermes_cli plugin modules."""
    home = tmp_path / "isolated-hermes-home"
    home.mkdir()
    token = set_hermes_home_override(home)
    original_import = builtins.__import__

    def reject_retired_owner(name, *args, **kwargs):
        if name == "hermes_cli.plugins_discovery":
            raise ModuleNotFoundError("retired plugin discovery owner", name=name)
        return original_import(name, *args, **kwargs)

    try:
        invalidate_plugin_skill_commands()
        with (
            patch("builtins.__import__", side_effect=reject_retired_owner),
            patch("plugin_runtime.lifecycle.discover_plugins") as discover,
            patch("plugin_runtime.lifecycle.get_plugin_manager") as get_manager,
            patch("agent.skill_utils.get_disabled_skill_names", return_value=set()),
            patch("plugin_runtime.discovery._get_disabled_plugins", return_value=set()),
        ):
            manager = get_manager.return_value
            manager.list_plugin_skill_metadata.return_value = []
            assert get_plugin_skill_commands() == {}
            discover.assert_called_once_with()
            manager.list_plugin_skill_metadata.assert_called_once_with()
    finally:
        invalidate_plugin_skill_commands()
        reset_hermes_home_override(token)
