"""Definitions of Exceptions used in the carps package."""

from __future__ import annotations


class NotSupportedError(Exception):
    """Generic exception for when a feature is not supported."""


class AskAndTellNotSupportedError(NotSupportedError):
    """Exception for when Ask and Tell is not supported."""
