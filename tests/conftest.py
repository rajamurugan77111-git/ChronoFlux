"""Test configuration: isolate users/secrets and disable login for the functional tests.
Auth itself is exercised in test_security.py by turning it back on."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="chronoflux-test-")
os.environ.setdefault("CHRONOFLUX_AUTH", "0")
os.environ.setdefault("CHRONOFLUX_USERS_DB", os.path.join(_tmp, "users.db"))
os.environ.setdefault("CHRONOFLUX_SECRET_KEY_PATH", os.path.join(_tmp, "secret_key"))
os.environ.setdefault("CHRONOFLUX_ADMIN_PASSWORD", "test-admin-pass")
