"""Python port of `packages/coding-agent/test/pi-user-agent.test.ts`."""

from __future__ import annotations

import platform
import re
import sys

from pi_coding_agent.utils.version_check import get_pi_user_agent


def test_formats_the_user_agent_expected_by_pi_dev():
    runtime = f"python/{platform.python_version()}"
    user_agent = get_pi_user_agent("1.2.3")

    assert user_agent == f"pi/1.2.3 ({sys.platform}; {runtime}; {platform.machine()})"
    assert re.match(r"^pi/[^\s()]+ \([^;()]+;\s*[^;()]+;\s*[^()]+\)$", user_agent)
