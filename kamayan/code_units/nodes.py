"""Utilities for managing a global tree of code units."""

import weakref
from collections.abc import Callable
from typing import Generic, Optional, Type, TypeVar

from kamayan.code_units.parameters import KamayanParams


class Node:
    """Tree for all code units."""

    def __init__(self, parent: Optional["Node"] = None):
        """Initialize the node."""
        self.parent: Optional["Node"] = None
        self.children: weakref.WeakValueDictionary[int, "Node"] = (
            weakref.WeakValueDictionary()
        )
        if parent:
            self.parent = parent
            self.parent.add_child(self)

    @property
    def child_ids(self) -> set[int]:
        """Get set a of child ids."""
        return {id for id in self.children.keys()}

    def _get_children(self) -> weakref.WeakValueDictionary[int, "Node"]:
        unique_ids = self.child_ids
        unique_children = self.children.copy()
        for child in self.children.values():
            child_descendents = child._get_children()
            child_descendents_ids = child.child_ids
            for c in child_descendents.values():
                child_descendents_ids = child_descendents_ids | c.child_ids

            unique_ids = unique_ids | child_descendents_ids
            unique_children.update({k: v for k, v in child_descendents.items()})

        return unique_children

    def get_children(self) -> list["Node"]:
        """Return set of all children."""
        return [v for v in self._get_children().values()]

    def add_child(self, node: "Node"):
        """Add a child."""
        self.children[id(node)] = node

    def set_params(self, params: KamayanParams) -> None:
        """Check our parameters for any potential conflicts.

        Used as a last chance to set a unit's runtime parameters one
        last time based on the global input parameters.

        Args:
            params: parameters that will be used for initialization
        """
        ...

    def pretty(self, level=0) -> str:
        """Formatted print of tree."""
        indent = "  " * level
        out = f"{indent}{self.__class__.__name__}\n"
        for child in list(self.children.values()):  # copy since it may shrink
            out += child.pretty(level + 1)
        return out


N = TypeVar("N", bound="Node")
T = TypeVar("T")
setter_interface = Callable[[T, N], None]


def _set_node(self: N, value: N):
    self.add_child(value)


class AutoProperty(Generic[T]):
    """Callable to generate getter/setters for child nodes."""

    def __init__(self, set_node: setter_interface = _set_node):
        """Initialize with the node setter."""
        self.set_node = set_node

    def __call__(self, n: Type[N], name: str):
        """Callable to generate getter/setters for child nodes."""

        def _getter(_self: T) -> N:
            return getattr(_self, "_" + name)

        def _setter(_self: T, value: N):
            setattr(_self, "_" + name, value)
            self.set_node(_self, value)

        return property(_getter, _setter)


auto_property_node = AutoProperty()
