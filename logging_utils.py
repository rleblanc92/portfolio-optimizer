import atexit
import logging
from logging.handlers import QueueHandler
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Lock, Thread

from tqdm.auto import tqdm


_logger = logging.getLogger('portfolio')
_logger.addHandler(logging.NullHandler())
_logger.propagate = False
_progress_stream = None
_progress_stream_lock = Lock()


class _AsyncProgressStream:
    def __init__(self, stream):
        self._stream = stream
        self._queue = Queue(maxsize=128)
        self._thread = Thread(target=self._write_queued, daemon=True)
        self._thread.start()

    def write(self, text):
        try:
            self._queue.put_nowait(text)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(text)
            except Full:
                pass
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return self._stream.isatty()

    def _write_queued(self):
        while True:
            text = self._queue.get()
            if text is None:
                return
            try:
                self._stream.write(text)
                self._stream.flush()
            except (OSError, ValueError):
                return

    def close(self):
        try:
            self._queue.put_nowait(None)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except Full:
                return
        self._thread.join(timeout=1.0)


def get_progress_stream():
    global _progress_stream
    stream = sys.stderr
    if not stream.isatty():
        return stream
    with _progress_stream_lock:
        if _progress_stream is None or _progress_stream._stream is not stream:
            _progress_stream = _AsyncProgressStream(stream)
            atexit.register(_progress_stream.close)
        return _progress_stream


class _OncePerWarning(logging.Filter):
    def __init__(self):
        super().__init__()
        self._seen = set()

    def filter(self, record):
        if record.levelno < logging.WARNING:
            return True
        key = (record.name, record.msg)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


class _TqdmStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def get_logger(name):
    return _logger.getChild(name)


def configure_logging():
    if any(getattr(handler, '_portfolio_handler', False) for handler in _logger.handlers):
        return _logger

    log_path = Path('outputs/logs/execution_debug.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_OncePerWarning())
    file_handler._portfolio_handler = True

    console_handler = _TqdmStreamHandler(stream=get_progress_stream())
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_OncePerWarning())
    console_handler._portfolio_handler = True

    _logger.setLevel(logging.INFO)
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    return _logger


def configure_worker_logging(queue):
    handler = QueueHandler(queue)
    handler.setLevel(logging.INFO)
    handler.addFilter(_OncePerWarning())
    _logger.handlers = [handler]
    _logger.setLevel(logging.INFO)