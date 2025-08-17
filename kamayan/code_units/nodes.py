"""Utilities for managing a global tree of code units."""

from typing import Optional, ValuesView

from kamayan.code_units.parameters import KamayanParams


class Node:
    """Tree for all code units."""

    _root: Optional["Node"] = None

    def __init__(self, parent: Optional["Node"] = None):
        """Initialize the node."""
        self.parent: Optional["Node"] = None
        self.children: dict[int, "Node"] = {}

        if not Node._root:
            Node._root = self
        else:
            self.parent = parent if parent else Node._root
            self.parent.add_child(self)

    @classmethod
    def root(cls) -> Optional["Node"]:
        """Get the root node."""
        return cls._root

    def _get_children(self) -> dict[int, "Node"]:
        children = self.children
        ids = children.keys()
        for child in self.children.values():
            children_children = child.children | child._get_children()
            new_ids = children_children.keys() - ids
            ids = ids | new_ids
            children = children | {id: children_children[id] for id in new_ids}

        return children

    def get_children(self) -> ValuesView["Node"]:
        """Return set of all children."""
        return self._get_children().values()

    def add_child(self, node: "Node") -> "Node":
        """Add a child."""
        if node.parent:
            # orphan the node from its parent
            # node.parent.children.pop(id(node))
            node.parent = self

        self.children[id(node)] = node
        return node

    def set_params(self, params: KamayanParams) -> None:
        """Set the input parameters for this node."""
        raise NotImplementedError("Node does not implement set_params")

    def finalize(self):
        """Clean up the tree."""
        for child in self.get_children():
            child.parent = None
            child.children = {}
