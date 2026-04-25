"""Tests for modernized registry."""

import pytest

from ludwig.utils.registry import Registry


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry()
        reg["foo"] = "bar"
        assert reg["foo"] == "bar"

    def test_register_decorator(self):
        reg = Registry()

        @reg.register("my_class")
        class MyClass:
            pass

        assert reg["my_class"] is MyClass

    def test_default_registration(self):
        reg = Registry()

        @reg.register("default_cls", default=True)
        class DefaultClass:
            pass

        assert reg[None] is DefaultClass
        assert reg["none"] is DefaultClass

    def test_unregister(self):
        reg = Registry()
        reg["foo"] = "bar"
        reg.unregister("foo")
        assert "foo" not in reg

    def test_unregister_missing_raises(self):
        reg = Registry()
        with pytest.raises(KeyError):
            reg.unregister("nonexistent")

    def test_get_default(self):
        reg = Registry()

        @reg.register("cls", default=True)
        class Cls:
            pass

        assert reg.get_default() is Cls

    def test_get_default_none(self):
        reg = Registry()
        reg["foo"] = "bar"
        assert reg.get_default() is None

    def test_list_registered(self):
        reg = Registry()
        reg["a"] = 1
        reg["b"] = 2

        @reg.register("c", default=True)
        class C:
            pass

        names = reg.list_registered()
        assert "a" in names
        assert "b" in names
        assert "c" in names
        assert None not in names

    def test_parent_delegation(self):
        parent = Registry()
        parent["parent_key"] = "parent_value"
        child = Registry(parent)
        child["child_key"] = "child_value"
        assert child["parent_key"] == "parent_value"
        assert child["child_key"] == "child_value"

    def test_contains(self):
        reg = Registry()
        reg["x"] = 1
        assert "x" in reg
        assert "y" not in reg

    def test_iteration(self):
        reg = Registry()
        reg["a"] = 1
        reg["b"] = 2
        assert set(reg.keys()) == {"a", "b"}
