import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def check(tmp_path, env_file: str) -> subprocess.CompletedProcess:
    (tmp_path / ".env").write_text(env_file)
    env = {k: v for k, v in os.environ.items() if k not in ("GROQ_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER")}
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "check_llm_key.py")], cwd=tmp_path, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("env_file, ok, named", [
    ("GROQ_API_KEY=g\n", True, ""),
    ("LLM_PROVIDER=openai\nOPENAI_API_KEY=o\n", True, ""),  # an OpenAI key alone is enough when OpenAI is chosen
    ("LLM_PROVIDER=openai\nGROQ_API_KEY=g\n", False, "OPENAI_API_KEY"),
    ("OPENAI_API_KEY=o\n", False, "GROQ_API_KEY"),
    ("LLM_PROVIDER=claude\nGROQ_API_KEY=g\n", False, "LLM_PROVIDER"),
])
def test_start_checks_the_key_of_the_chosen_provider(tmp_path, env_file, ok, named):
    result = check(tmp_path, env_file)
    assert (result.returncode == 0) is ok and named in result.stderr
