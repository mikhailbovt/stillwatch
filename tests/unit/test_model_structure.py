import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from stillwatch.perception import _restore_locked_architecture


class Identity:
    def __call__(self, value):
        return value


class SiLU:
    pass


class Graph:
    """Two paths share a constructor activation, as in the source regression."""

    def __init__(self):
        activation = SiLU()
        self.first = SimpleNamespace(act=activation)
        self.second = SimpleNamespace(act=activation)

    def get_submodule(self, path):
        return getattr(self, path)

    def named_modules(self, remove_duplicate=True):
        assert not remove_duplicate
        return [
            ("", self),
            ("first", self.first),
            ("first.act", self.first.act),
            ("second", self.second),
            ("second.act", self.second.act),
        ]


@pytest.fixture
def graph(monkeypatch):
    # The graph contract does not require the optional inference dependencies.
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(nn=SimpleNamespace(Identity=Identity, SiLU=SiLU))
    )
    return Graph()


def source_entry(first_activation="Identity", second_activation="SiLU"):
    modules = {
        "": f"{__name__}.Graph",
        "first": "types.SimpleNamespace",
        "first.act": f"{__name__}.{first_activation}",
        "second": "types.SimpleNamespace",
        "second.act": f"{__name__}.{second_activation}",
    }
    return {
        "module_types_sha256": hashlib.sha256(
            json.dumps(modules, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "identity_activations": ["first.act"],
    }


def test_restores_source_activation_without_changing_shared_constructor_peer(graph):
    _restore_locked_architecture(graph, source_entry())
    assert graph.first.act(-2) == -2
    assert type(graph.second.act) is SiLU


def test_unchanged_graph_needs_no_override(graph):
    entry = source_entry(first_activation="SiLU")
    entry.pop("identity_activations")
    original = graph.first.act
    _restore_locked_architecture(graph, entry)
    assert graph.first.act is graph.second.act is original


def test_already_restored_source_graph_is_accepted(graph):
    graph.first.act = Identity()
    _restore_locked_architecture(graph, source_entry())


def test_checks_shared_activation_paths_individually(graph):
    graph.second.act = Identity()
    with pytest.raises(ValueError, match="architecture"):
        _restore_locked_architecture(graph, source_entry())


@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_missing_or_changed_source_graph_is_rejected(graph, digest):
    entry = source_entry()
    entry["module_types_sha256"] = digest
    with pytest.raises(ValueError, match="architecture"):
        _restore_locked_architecture(graph, entry)


def test_override_cannot_replace_an_arbitrary_module_attribute(graph):
    entry = source_entry()
    entry["identity_activations"] = ["first.parameter"]
    graph.first.parameter = SiLU()
    with pytest.raises(ValueError, match="Unsupported"):
        _restore_locked_architecture(graph, entry)
    assert type(graph.first.parameter) is SiLU


def test_override_cannot_silently_replace_an_unexpected_module_type(graph):
    graph.first.act = object()
    with pytest.raises(ValueError, match="Unsupported"):
        _restore_locked_architecture(graph, source_entry())
