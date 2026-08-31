"""Runtime settings: where Qwen lives and what the harness is allowed to do.

Precedence, lowest to highest: config/default.yaml, FLUFC_* environment
variables, then explicit keyword overrides (which is how the CLI flags get in).
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "config" / "default.yaml"

# The Slurm job that `~/.local/bin/qwen` submits, and the port it serves on.
JOB_NAME = "qwen-vllm"
DEFAULT_PORT = 8000


def _slurm_node():
    """Node running our vLLM job, or None if there isn't one."""
    if not shutil.which("squeue"):
        return None
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-n", JOB_NAME,
             "-o", "%N"],
            capture_output=True, text=True, timeout=10,
        ).stdout.split()
    except (subprocess.SubprocessError, OSError):
        return None
    return out[0] if out else None


def discover_endpoint():
    """Base URL of the vLLM API: $QWEN_URL, else the job's node, else local."""
    url = os.environ.get("QWEN_URL")
    if url:
        return url.rstrip("/")
    node = _slurm_node()
    host = node if node else "localhost"
    return f"http://{host}:{DEFAULT_PORT}/v1"


def _load_yaml(path):
    """Read the handful of scalar keys we keep in default.yaml.

    Deliberately not a YAML parser: keeping the core stdlib-only means the
    harness runs on a bare login node with no environment to activate.
    """
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([a-z_]+):\s*(.*)$", line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        values[key] = _coerce(raw)
    return values


def _coerce(raw):
    """Turn a scalar string from YAML or the environment into a Python value."""
    if raw in ("", "null", "~"):
        return None
    if raw in ("true", "false"):
        return raw == "true"
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d*\.\d+", raw):
        return float(raw)
    return raw.strip("'\"")


@dataclass
class Settings:
    # model
    endpoint: str = ""
    model: str = "qwen3.8-27b"
    temperature: float = 0.3
    max_tokens: int = 4096
    request_timeout: int = 600
    tool_mode: str = "auto"          # auto | prompt | native

    # agent loop
    max_steps: int = 16
    max_tool_output_chars: int = 6000
    context_budget: int = 28000

    # tools
    search_backend: str = "duckduckgo"
    search_results: int = 6
    cache_ttl: int = 21600

    # paths
    cache_dir: Path = ROOT / "data" / "cache"
    output_dir: Path = ROOT / "outputs"
    archive_dir: Path = ROOT / "forecasts"     # committed; the public record
    site_dir: Path = ROOT / "docs"             # what GitHub Pages serves

    @classmethod
    def load(cls, **overrides):
        """Build settings from default.yaml + FLUFC_* env vars + overrides."""
        known = {f.name: f for f in fields(cls)}
        values = {k: v for k, v in _load_yaml(CONFIG_FILE).items() if k in known}

        for name in known:
            env = os.environ.get("FLUFC_" + name.upper())
            if env is not None:
                values[name] = _coerce(env)

        values.update({k: v for k, v in overrides.items()
                       if k in known and v is not None})

        for name in ("cache_dir", "output_dir", "archive_dir", "site_dir"):
            if name in values and values[name] is not None:
                values[name] = Path(values[name])

        settings = cls(**values)
        if not settings.endpoint:
            settings.endpoint = discover_endpoint()
        settings.endpoint = settings.endpoint.rstrip("/")
        return settings

    def run_dir(self, stamp):
        """Directory for one agent run's transcript and artifacts."""
        return self.output_dir / "runs" / stamp


# The settings the tools use. The CLI installs its own (flags applied) with
# use(); anything else gets the defaults on first ask.
_ACTIVE = None


def active():
    """The process-wide settings, loaded on first use."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = Settings.load()
    return _ACTIVE


def use(settings):
    """Install settings for the tools to pick up."""
    global _ACTIVE
    _ACTIVE = settings
    return settings
