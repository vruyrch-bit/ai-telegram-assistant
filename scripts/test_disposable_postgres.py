"""Run pytest against a newly created UTF-8 cluster, never the application DB.

Usage: venv/bin/python scripts/test_disposable_postgres.py [pytest arguments]
The cluster listens only on its private /tmp Unix socket and is stopped on exit.
"""
import getpass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote


def main():
    repo = Path(__file__).resolve().parents[1]
    candidates = sorted(Path('/usr/lib/postgresql').glob('*/bin/initdb'),
                        key=lambda path: int(path.parent.parent.name), reverse=True)
    initdb = shutil.which('initdb') or (str(candidates[0]) if candidates else None)
    if not initdb:
        raise SystemExit('Local PostgreSQL server tools are required.')
    binaries = Path(initdb).parent
    cluster = Path(tempfile.mkdtemp(prefix='bot-validation-'))
    data = cluster / 'data'
    started = False
    try:
        with (cluster / 'setup.log').open('w') as log:
            subprocess.run([initdb, '-D', str(data), '-A', 'trust', '--no-locale', '-E', 'UTF8'],
                           check=True, stdout=log, stderr=log)
            subprocess.run([str(binaries / 'pg_ctl'), '-D', str(data), '-l', str(cluster / 'server.log'),
                            '-o', f"-h '' -k {cluster} -p 55439", '-w', 'start'],
                           check=True, stdout=log, stderr=log)
            started = True
            subprocess.run([str(binaries / 'createdb'), '-h', str(cluster), '-p', '55439',
                            'bot_test_validation'], check=True, stdout=log, stderr=log)
        url = f'postgresql://{quote(getpass.getuser())}@/bot_test_validation?host={quote(str(cluster), safe="")}&port=55439'
        env = dict(os.environ, BOT_TEST_DATABASE_URL=url, PYTHON_DOTENV_DISABLED='1')
        env['PATH'] = str(binaries) + os.pathsep + env.get('PATH', '')
        print('Using a disposable local PostgreSQL cluster.', flush=True)
        return subprocess.run([sys.executable, '-m', 'pytest', *(sys.argv[1:] or ['-q'])],
                              cwd=repo, env=env).returncode
    finally:
        if started:
            subprocess.run([str(binaries / 'pg_ctl'), '-D', str(data), '-m', 'fast', '-w', 'stop'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # Exact directory created by mkdtemp above, containing only test data.
        shutil.rmtree(cluster)


if __name__ == '__main__':
    raise SystemExit(main())
