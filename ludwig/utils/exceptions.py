# -*- coding: utf-8 -*-

from __future__ import unicode_literals


class LudwigException(Exception):
    """Raised when an exception is encountered when running a Ludwig job."""


class LudwigRateLimitException(LudwigException):
    """Raised when an io request has been rate limited"""


class LudwigOptionalDependencyException(LudwigException):
    """Raised when an exception is encountered when an optional plugin is
    missing."""


def missing_dependency_generator(package, dep):
    def missing_dep():
        raise LudwigOptionalDependencyException(
            "The {package} optional dependency is missing. "
            "Please run pip install ludwig[{dep}] to install this "
            "dependency".format(
                package=package, dep=dep
            )
        )
        return missing_dep
