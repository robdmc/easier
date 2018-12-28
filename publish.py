import subprocess
"""
NOTE:  As of 12/28/2018, pypi is recommending twine as uploader

Here is the doc page for that.  I just haven't had the time to do it.
https://pypi.org/project/twine/
"""

subprocess.call('pip install wheel'.split())
subprocess.call('python setup.py clean --all'.split())
subprocess.call('python setup.py sdist'.split())
# subprocess.call('pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz'.split())
subprocess.call('python setup.py register sdist bdist_wheel upload'.split())
