"""Pytest unit tests for Node tree infrastructure."""

import pytest
import weakref

from kamayan.code_units.nodes import Node, AutoProperty, auto_property_node
from kamayan.code_units.parameters import KamayanParams


class TestNode:
    """Test suite for Node base class."""

    def test_create_node_without_parent(self):
        """Test creating a Node without a parent."""
        node = Node()
        assert node is not None
        assert node.parent is None

    def test_create_node_with_parent(self):
        """Test creating a Node with a parent."""
        parent = Node()
        child = Node(parent=parent)

        assert child.parent is parent
        assert id(child) in parent.child_ids

    def test_add_child(self):
        """Test manually adding a child to a node."""
        parent = Node()
        child = Node()

        parent.add_child(child)

        assert id(child) in parent.child_ids
        children = parent.get_children()
        assert child in children

    def test_add_multiple_children(self):
        """Test adding multiple children to a node."""
        parent = Node()
        child1 = Node()
        child2 = Node()
        child3 = Node()

        parent.add_child(child1)
        parent.add_child(child2)
        parent.add_child(child3)

        children = parent.get_children()
        assert len(children) == 3
        assert child1 in children
        assert child2 in children
        assert child3 in children

    def test_child_ids_property(self):
        """Test child_ids property returns correct ids."""
        parent = Node()
        child1 = Node()
        child2 = Node()

        parent.add_child(child1)
        parent.add_child(child2)

        child_ids = parent.child_ids
        assert id(child1) in child_ids
        assert id(child2) in child_ids
        assert len(child_ids) == 2

    def test_get_children_returns_list(self):
        """Test get_children returns a list of Node objects."""
        parent = Node()
        child1 = Node()
        child2 = Node()

        parent.add_child(child1)
        parent.add_child(child2)

        children = parent.get_children()
        assert isinstance(children, list)
        assert len(children) == 2

    def test_nested_children(self):
        """Test getting children recursively from nested structure."""
        root = Node()
        child1 = Node()
        child2 = Node()
        grandchild1 = Node()
        grandchild2 = Node()

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild1)
        child2.add_child(grandchild2)

        # get_children should return all descendants
        all_children = root.get_children()
        assert len(all_children) == 4
        assert child1 in all_children
        assert child2 in all_children
        assert grandchild1 in all_children
        assert grandchild2 in all_children

    def test_weak_references(self):
        """Test that children use weak references (WeakValueDictionary)."""
        parent = Node()
        child = Node()
        parent.add_child(child)

        # Verify child is in children
        assert id(child) in parent.child_ids

        # Delete child reference
        child_id = id(child)
        del child

        # WeakValueDictionary should have removed the entry
        # Note: This may not be immediate; gc might need to run
        import gc

        gc.collect()
        assert child_id not in parent.child_ids

    def test_set_params_not_implemented(self):
        """Test that base Node raises NotImplementedError for set_params."""
        node = Node()

        with pytest.raises(NotImplementedError):
            node.set_params(None)

    def test_pretty_print_single_node(self):
        """Test pretty printing a single node."""
        node = Node()
        output = node.pretty()

        assert "Node" in output
        assert output.strip() == "Node"

    def test_pretty_print_with_children(self):
        """Test pretty printing a node tree."""
        parent = Node()
        child1 = Node()
        child2 = Node()

        parent.add_child(child1)
        parent.add_child(child2)

        output = parent.pretty()

        assert "Node" in output
        # Should have indentation for children
        assert "  Node" in output
        # Count number of "Node" occurrences (should be 3)
        assert output.count("Node") == 3


class ConcreteNode(Node):
    """Concrete Node implementation for testing."""

    def __init__(self, name, parent=None):
        """Initialize with a name."""
        super().__init__(parent)
        self.name = name
        self.params_set = False

    def set_params(self, params: KamayanParams) -> None:
        """Implement set_params for testing."""
        self.params_set = True


class TestConcreteNode:
    """Test Node with concrete implementation."""

    def test_concrete_node_set_params(self):
        """Test that concrete implementation can override set_params."""
        node = ConcreteNode("test")
        assert not node.params_set

        node.set_params(None)
        assert node.params_set

    def test_pretty_print_concrete_nodes(self):
        """Test pretty printing with custom node types."""
        parent = ConcreteNode("parent")
        child1 = ConcreteNode("child1")
        child2 = ConcreteNode("child2")

        parent.add_child(child1)
        parent.add_child(child2)

        output = parent.pretty()

        assert "ConcreteNode" in output
        assert output.count("ConcreteNode") == 3


class TestAutoProperty:
    """Test suite for AutoProperty descriptor."""

    def test_create_auto_property(self):
        """Test creating an AutoProperty instance."""
        auto_prop = AutoProperty()
        assert auto_prop is not None

    def test_auto_property_creates_property(self):
        """Test that AutoProperty creates a property descriptor."""

        class TestClass(Node):
            child_node = auto_property_node(Node, "child_node")

            def __init__(self):
                super().__init__()
                self._child_node = Node()

        obj = TestClass()
        # Access via property
        child = obj.child_node
        assert isinstance(child, Node)

    def test_auto_property_setter(self):
        """Test that AutoProperty setter works."""

        class TestClass(Node):
            child_node = auto_property_node(Node, "child_node")

            def __init__(self):
                super().__init__()
                self._child_node = None

        obj = TestClass()
        new_child = Node()

        # Set via property
        obj.child_node = new_child

        # Verify it was set
        assert obj._child_node is new_child
        assert obj.child_node is new_child

    def test_auto_property_adds_child_to_tree(self):
        """Test that AutoProperty setter adds child to parent's tree."""

        class TestClass(Node):
            child_node = auto_property_node(Node, "child_node")

            def __init__(self):
                super().__init__()
                self._child_node = None

        parent = TestClass()
        child = Node()

        # Setting via property should add to tree
        parent.child_node = child

        # Verify child is in parent's children
        assert id(child) in parent.child_ids

    def test_auto_property_node_global_instance(self):
        """Test using the global auto_property_node instance."""

        class ParentNode(Node):
            nested = auto_property_node(Node, "nested")

            def __init__(self):
                super().__init__()
                self._nested = Node()

        parent = ParentNode()
        assert isinstance(parent.nested, Node)


class TestAutoPropertyIntegration:
    """Integration tests for AutoProperty with Node trees."""

    def test_multiple_auto_properties(self):
        """Test a class with multiple auto properties."""

        class MultiPropNode(Node):
            prop1 = auto_property_node(Node, "prop1")
            prop2 = auto_property_node(Node, "prop2")
            prop3 = auto_property_node(Node, "prop3")

            def __init__(self):
                super().__init__()
                self._prop1 = None
                self._prop2 = None
                self._prop3 = None

        node = MultiPropNode()

        # Set all properties
        child1 = Node()
        child2 = Node()
        child3 = Node()

        node.prop1 = child1
        node.prop2 = child2
        node.prop3 = child3

        # Verify all added to tree
        children = node.get_children()
        assert len(children) == 3
        assert child1 in children
        assert child2 in children
        assert child3 in children

    def test_nested_auto_properties(self):
        """Test auto properties with nested node structures."""

        class NestedNode(Node):
            child = auto_property_node(Node, "child")

            def __init__(self):
                super().__init__()
                self._child = None

        root = NestedNode()
        level1 = NestedNode()
        level2 = NestedNode()

        root.child = level1
        level1.child = level2

        # Check tree structure
        all_children = root.get_children()
        assert level1 in all_children
        assert level2 in all_children

    def test_auto_property_with_concrete_nodes(self):
        """Test auto properties with concrete Node implementations."""

        class ParentType(Node):
            child = auto_property_node(ConcreteNode, "child")

            def __init__(self):
                super().__init__()
                self._child = None

        parent = ParentType()
        child = ConcreteNode("test_child")

        parent.child = child

        assert parent.child.name == "test_child"
        assert id(child) in parent.child_ids
