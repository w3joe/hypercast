"""Loopback backend discovery and single-owner workspace locking (macOS/Linux)."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

CLI_PROTOCOL = 1


def request(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url + path, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read().decode()
            return json.loads(raw) if 'json' in response.headers.get('Content-Type', '') else raw
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            raw = json.loads(raw).get('detail', raw)
        except ValueError:
            pass
        raise ValueError(str(raw)) from error


def identity(project, root):
    from importlib.metadata import version
    from .graph_architecture import REVISION
    return {'cli_protocol': CLI_PROTOCOL, 'project_root': str(Path(project).resolve()),
            'results_root': str(Path(root).resolve()), 'pid': os.getpid(),
            'version': version('hypercast4d'), 'graph_revision': REVISION}


@contextmanager
def workspace_owner(project, root, url=None):
    directory = Path(root).resolve() / '.runtime'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'owner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another backend owns this results workspace; reuse it instead.') from error
        record = directory / 'server.json'
        if url:
            temporary = directory / f'server-{os.getpid()}.tmp'
            temporary.write_text(json.dumps({**identity(project, root), 'url': url}))
            temporary.replace(record)
        try:
            yield
        finally:
            if url and record.exists():
                try:
                    if json.loads(record.read_text()).get('pid') == os.getpid():
                        record.unlink()
                except (OSError, ValueError):
                    pass
            fcntl.flock(lock, fcntl.LOCK_UN)


def discover(project, root):
    path = Path(root) / '.runtime/server.json'
    if not path.exists():
        return None
    try:
        saved = json.loads(path.read_text())
        url = saved['url']
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'}:
            raise RuntimeError('Invalid backend discovery address')
        with urllib.request.urlopen(url + '/api/v1/identity', timeout=1) as response:
            live = json.load(response)
    except (OSError, ValueError, KeyError):
        return None
    expected = identity(project, root)
    if any(live.get(key) != expected[key] for key in ('cli_protocol', 'project_root', 'results_root', 'version', 'graph_revision')):
        raise RuntimeError('Backend version or workspace mismatch; stop the old backend explicitly.')
    if live.get('pid') != saved.get('pid'):
        raise RuntimeError('Backend identity changed; inspect the workspace runtime record.')
    return {**live, 'url': url}


def ensure_backend(project, root, port=0):
    found = discover(project, root)
    if found:
        return found
    # A pre-CLI playground cannot participate in locking. Never start a second
    # worker against its default workspace, even if it has no discovery record.
    if Path(root).resolve() == (Path(project).resolve() / 'results/playground'):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=.5) as response:
                if json.load(response).get('ok'):
                    raise RuntimeError('An unregistered playground is running on 8765. Stop it and restart with the updated package first.')
        except (OSError, ValueError):
            pass
    directory = Path(root) / '.runtime'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'server.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-m', 'hypercast4d.workspace_runtime',
            str(Path(project).resolve()), str(Path(root).resolve()), str(port)],
            cwd=project, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        found = discover(project, root)
        if found:
            return found
        if process.poll() is not None:
            # Another simultaneous launcher may have won the workspace lock.
            for _ in range(20):
                time.sleep(.1)
                found = discover(project, root)
                if found:
                    return found
            raise RuntimeError(f'Backend failed to start; inspect {directory / "server.log"}')
        time.sleep(.1)
    raise RuntimeError(f'Backend startup timed out; inspect {directory / "server.log"}')


def stop_backend(found):
    active = [job for job in request(found['url'], '/api/v1/jobs')
              if job['status']['state'] in {'queued', 'starting', 'running'}]
    if active:
        raise ValueError('Backend has active jobs. Cancel them explicitly before stopping it.')
    os.kill(found['pid'], signal.SIGTERM)


def serve(project, root, port=0, legacy=None):
    import uvicorn
    from .playground import create_app
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', port))
    url = f'http://127.0.0.1:{sock.getsockname()[1]}'
    app = create_app(Path(root), Path(project), Path(legacy or 'results/evaluation'), server_url=url)
    try:
        uvicorn.Server(uvicorn.Config(app, log_level='info')).run(sockets=[sock])
    finally:
        sock.close()


if __name__ == '__main__':
    serve(sys.argv[1], sys.argv[2], int(sys.argv[3]))
