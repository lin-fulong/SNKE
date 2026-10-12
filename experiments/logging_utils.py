import sys
from pathlib import Path


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


_LOG_FILE_HANDLE = None


def install_run_logger(run_dir: Path, prefix: str):
    global _LOG_FILE_HANDLE
    if _LOG_FILE_HANDLE is not None:
        return

    log_path = run_dir / "run.log"
    _LOG_FILE_HANDLE = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, _LOG_FILE_HANDLE)
    sys.stderr = TeeStream(sys.stderr, _LOG_FILE_HANDLE)
    print(f"[{prefix}] Logging stdout/stderr to {log_path}")
