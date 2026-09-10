"""Create a private PostgreSQL custom-format backup; never prints the DSN.

Usage: DATABASE_URL=... python scripts/backup_database.py /secure/path/backup.dump
Requires pg_dump. Store backups outside the repository; encrypt off-site copies.
"""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    url = os.environ.get('DATABASE_URL')
    if not url:
        parser.error('Set DATABASE_URL in the environment.')
    from psycopg.conninfo import conninfo_to_dict
    mapping = {'host': 'PGHOST', 'hostaddr': 'PGHOSTADDR', 'port': 'PGPORT',
               'dbname': 'PGDATABASE', 'user': 'PGUSER', 'password': 'PGPASSWORD',
               'sslmode': 'PGSSLMODE', 'sslcert': 'PGSSLCERT', 'sslkey': 'PGSSLKEY',
               'sslrootcert': 'PGSSLROOTCERT', 'sslcrl': 'PGSSLCRL',
               'connect_timeout': 'PGCONNECT_TIMEOUT', 'options': 'PGOPTIONS',
               'application_name': 'PGAPPNAME', 'channel_binding': 'PGCHANNELBINDING',
               'target_session_attrs': 'PGTARGETSESSIONATTRS'}
    try:
        params = conninfo_to_dict(url)
    except Exception:
        parser.error('DATABASE_URL is not a valid PostgreSQL connection string.')
    if set(params) - mapping.keys():
        parser.error('DATABASE_URL contains unsupported options; use pg_dump with a private service file.')
    env = {key: value for key, value in os.environ.items() if not key.startswith('PG')}
    env.update({mapping[key]: value for key, value in params.items()})
    # Refuse overwrites; mode 0600 protects the personal data in the backup.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as output:
        result = subprocess.run(['pg_dump', '--format=custom', '--no-owner', '--no-acl'],
                                stdout=output, stderr=subprocess.PIPE, env=env)
    if result.returncode:
        # Do not echo pg_dump stderr; it can contain connection details.
        raise SystemExit('Backup failed. The output is incomplete; check database access and pg_dump availability.')
    print('Backup created successfully.')


if __name__ == '__main__':
    main()
