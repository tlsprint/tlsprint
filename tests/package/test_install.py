"""Verify that the package can be installed and the metadata is set."""

import unittest
import venv
import tempfile
import pathlib
import shlex
import subprocess
import sys
from email.parser import HeaderParser

from . import __version__
from . import PACKAGE_NAME


def checked_subprocess_run(command):
    """Run the command, print the output, and check if the call was successful.

    Using this wrapper function in the tests offers the follow functionality:

    -   Pass the command as a string, instead of a list. This is easier to
        write, and `shlex.split` ensure it is split correctly.
    -   Capture and decode both stdout and stderr.
    -   Print both stdout and stderr, to include the output in the test case.
    -   Raise an exception on a non-zero exit status.
    """
    args = shlex.split(command)
    completed = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out = completed.stdout.decode()
    err = completed.stderr.decode()

    # Print the subprocess output to include in the test output
    print(out, file=sys.stdout)
    print(err, file=sys.stderr)

    # After printing the output, raise an exception on a non-zero exit status.
    completed.check_returncode()

    return out, err


class TestInstall(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory and initialize a virtual
        environment."""
        self.tempdir = tempfile.TemporaryDirectory()
        self.tempdir_path = pathlib.Path(self.tempdir.name)
        self.python = self.tempdir_path / "bin" / "python"
        venv.create(
            env_dir=self.tempdir_path, system_site_packages=False, with_pip=True
        )

        # Ensure the virtual environment has a recent version of pip which
        # has support for PEP 517.
        checked_subprocess_run(f"{self.python} -m pip install --upgrade pip")

    def test_install(self):
        """Install the current project in the virtual environment."""
        # This call should not throw an exception
        checked_subprocess_run(f"{self.python} -m pip install .")

        # Check the version number from `pip info`
        info, _ = checked_subprocess_run(f"{self.python} -m pip show {PACKAGE_NAME}")

        # The info section from pip is formatted as a RFC 2882 mail header.
        parser = HeaderParser()
        data = parser.parsestr(info)
        version = data["version"]

        # Version should be set, should not be the default 0.0.0, and should
        # match __version__ set in the package.
        self.assertTrue(version)
        self.assertNotEqual(version, "0.0.0")
        self.assertEqual(version, __version__)

    def tearDown(self):
        """Delete temporary directory."""
        self.tempdir.cleanup()
