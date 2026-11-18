"""An OS-released local GPU-study lock; no stale PID guessing or process killing."""
from contextlib import contextmanager
import json
import os
from .records import now


@contextmanager
def study_lock(directory):
    directory.mkdir(parents=True,exist_ok=True)
    stream=(directory/'execution.lock').open('a+b')
    stream.seek(0,2)
    if stream.tell()==0:stream.write(b'0');stream.flush()
    stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError('Another process owns this study; do not launch duplicate training') from exc
    try:
        stream.seek(1);stream.truncate()
        stream.write(json.dumps({'pid':os.getpid(),'at':now()}).encode());stream.flush()
        yield
    finally:
        stream.seek(0)
        if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()
