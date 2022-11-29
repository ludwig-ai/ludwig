_logged = set()


def log_once(key: str) -> bool:
    """Returns True if this is the "first" call for a given key.

    Example:
        if log_once("some_key"):
            logger.info("Some verbose logging statement") # noqa
    """

    if key not in _logged:
        _logged.add(key)
        return True
    return False
