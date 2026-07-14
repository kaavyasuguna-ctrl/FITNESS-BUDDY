#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
from functools import wraps
from typing import Any, Callable


class GlobalHttpxSettings:
    """Holds global state for all requests made using the `httpx` library."""

    verify: bool | str | None = None
    proxies: dict[str, str] | None = None

    @classmethod
    def get_verify_from_environment(cls) -> bool | str | None:
        """
        Get the verify value implied by environment variables and global state.
        Prioritizes environment variable over global state.
        """

        match os.environ.get("WX_CLIENT_VERIFY_REQUESTS"):
            case "True" | "":
                # Empty string means True (default verification)
                return True
            case "False":
                return False
            case None:
                return cls.verify
            case _ as env_verify:
                return env_verify

    @classmethod
    def get_effective_verify(cls) -> bool | str:
        """
        Get the effective verify value from environment variable and global state.
        Prioritizes environment variable over global state.
        Defaults to True if none are set.
        Returns the verify value to use for SSL verification.
        """

        if (verify := cls.get_verify_from_environment()) is not None:
            return verify

        return True

    @classmethod
    def inject_settings(cls, func: Callable) -> Callable:
        """
        Injects global httpx settings - such as proxies - to the keyword arguments.
        """

        @wraps(func)
        def wrapper(*args: Any, **kw: Any) -> Any:
            kwargs = {"proxies": cls.proxies}
            kwargs.update(kw)
            return func(*args, **kwargs)

        return wrapper
