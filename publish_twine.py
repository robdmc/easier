import subprocess
"""
NOTE:  As of 12/28/2018, pypi is recommending twine as uploader

Here is the doc page for that.  I just haven't had the time to do it.
https://pypi.org/project/twine/
"""

subprocess.call('pip install wheel'.split())
subprocess.call('pip install twine'.split())
subprocess.call('pwd'.split())
subprocess.call('rm -rf ./build'.split())
subprocess.call('rm -rf ./dist/'.split())
subprocess.call('python setup.py clean --all'.split())
subprocess.call('python setup.py sdist bdist_wheel'.split())
# subprocess.call('pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz'.split())
# subprocess.call('python setup.py register sdist bdist_wheel upload'.split())
subprocess.call('twine upload dist/*'.split())
