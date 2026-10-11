"""Deferred platform retry remains exposed by the canonical runtime manager."""

from plugin_runtime.manager import PluginManager


def test_plugin_manager_exposes_failed_platform_rearm():
    assert callable(getattr(PluginManager, "rearm_failed_platform", None))
