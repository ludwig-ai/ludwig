from typing import Optional


def PublicAPI(*args, **kwargs):
    """Annotation for documenting public APIs. Public APIs are classes and methods exposed to end users of Ludwig.

    If stability="stable", the APIs will remain backwards compatible across minor Ludwig releases
    (e.g., Ludwig 0.6 -> Ludwig 0.7).

    If stability="experimental", the APIs can be used by advanced users who are tolerant to and expect
    breaking changes. This will likely be seen in the case of incremental new feature development.

    Args:
        stability: One of {"stable", "experimental"}

    Examples:
        >>> from api_annotations import PublicAPI
        >>> @PublicAPI
        ... def func1(x):
        ...     return x
        >>> @PublicAPI(stability="experimental")
        ... def func2(y):
        ...     return y
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return PublicAPI(stability="stable")(args[0])

    if "stability" in kwargs:
        stability = kwargs["stability"]
        assert stability in ["stable", "experimental"], stability
    elif kwargs:
        raise ValueError(f"Unknown kwargs: {kwargs.keys()}")
    else:
        stability = "stable"

    def wrap(obj):
        if stability == "experimental":
            message = f"PublicAPI ({stability}): This API is {stability} and may change before becoming stable."
        else:
            message = "PublicAPI: This API is stable across Ludwig releases."

        _append_doc(obj, message=message)
        _mark_annotated(obj)
        return obj

    return wrap


def DeveloperAPI(*args, **kwargs):
    """Annotation for documenting developer APIs. Developer APIs are lower-level methods explicitly exposed to
    advanced Ludwig users and library developers. Their interfaces may change across minor Ludwig releases (for
    e.g., Ludwig 0.6.1 and Ludwig 0.6.2).

    Examples:
        >>> from api_annotations import DeveloperAPI
        >>> @DeveloperAPI
        ... def func(x):
        ...     return x
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return DeveloperAPI()(args[0])

    def wrap(obj):
        _append_doc(obj, message="DeveloperAPI: This API may change across minor Ludwig releases.")
        _mark_annotated(obj)
        return obj

    return wrap


def Deprecated(*args, **kwargs):
    """Annotation for documenting a deprecated API. Deprecated APIs may be removed in future releases of Ludwig
    (e.g., Ludwig 0.7 to Ludwig 0.8).

    Args:
        message: A message to help users understand the reason for the deprecation, and provide a migration path.

    Examples:
        >>> from api_annotations import Deprecated
        >>> @Deprecated
        ... def func(x):
        ...     return x
        >>> @Deprecated(message="g() is deprecated because the API is error prone. Please call h() instead.")
        ... def g(y):
        ...     return y
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return Deprecated()(args[0])

    message = "**DEPRECATED:** This API is deprecated and may be removed in a future Ludwig release."

    if "message" in kwargs:
        message += " " + kwargs["message"]
        del kwargs["message"]

    if kwargs:
        raise ValueError(f"Unknown kwargs: {kwargs.keys()}")

    def inner(obj):
        _append_doc(obj, message=message, directive="warning")
        _mark_annotated(obj)
        return obj

    return inner


def _append_doc(obj, message: str, directive: Optional[str] = None) -> str:
    """
    Args:
        message: An additional message to append to the end of docstring for a class
                 or method that uses one of the API annotations
        directive: A shorter message that provides contexts for the message and indents it.
                For example, this could be something like 'warning' or 'info'.
    """
    if not obj.__doc__:
        obj.__doc__ = ""

    obj.__doc__ = obj.__doc__.rstrip()

    indent = _get_indent(obj.__doc__)
    obj.__doc__ += "\n\n"
    if directive is not None:
        obj.__doc__ += f"{' ' * indent}.. {directive}::\n"
        obj.__doc__ += f"{' ' * (indent + 4)}{message}"
    else:
        obj.__doc__ += f"{' ' * indent}{message}"
    obj.__doc__ += f"\n{' ' * indent}"


def _mark_annotated(obj) -> None:
    # Set magic token for check_api_annotations linter.
    if hasattr(obj, "__name__"):
        obj._annotated = obj.__name__


def _is_annotated(obj) -> bool:
    # Check the magic token exists and applies to this class (not a subclass).
    return hasattr(obj, "_annotated") and obj._annotated == obj.__name__


def _get_indent(docstring: str) -> int:
    """
    Example:
        >>> def f():
        ...     '''Docstring summary.'''
        >>> f.__doc__
        'Docstring summary.'
        >>> _get_indent(f.__doc__)
        0
        >>> def g(foo):
        ...     '''Docstring summary.
        ...
        ...     Args:
        ...         foo: Does bar.
        ...     '''
        >>> g.__doc__
        'Docstring summary.\\n\\n    Args:\\n        foo: Does bar.\\n    '
        >>> _get_indent(g.__doc__)
        4
        >>> class A:
        ...     def h():
        ...         '''Docstring summary.
        ...
        ...         Returns:
        ...             None.
        ...         '''
        >>> A.h.__doc__
        'Docstring summary.\\n\\n        Returns:\\n            None.\\n        '
        >>> _get_indent(A.h.__doc__)
        8
    """
    if not docstring:
        return 0

    non_empty_lines = list(filter(bool, docstring.splitlines()))
    if len(non_empty_lines) == 1:
        # Docstring contains summary only.
        return 0

    # The docstring summary isn't indented, so check the indentation of the second non-empty line.
    return len(non_empty_lines[1]) - len(non_empty_lines[1].lstrip())
