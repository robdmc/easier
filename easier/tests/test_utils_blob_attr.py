from ..utils import BlobMixin, BlobAttr
import pytest

from ..utils import BlobMixin, BlobAttr
m0 = 1
m1 = {'one': 1}
m2 = {'sub': {'one': 1}}

class Param(BlobMixin):
    mut0 = BlobAttr(m0)
    mut1 = BlobAttr(m1)
    mut2 = BlobAttr(m2)

def test_deep_defaults_dont_mutate():
    p = Param()
    p.mut0 = 2
    p.mut1['one'] = 2
    p.mut2['sub']['one'] = 2
    assert p.mut0 == 2
    assert p.mut1['one'] == 2
    assert p.mut2['sub']['one'] == 2
    assert m0 == 1
    assert m1['one'] == 1
    assert m2['sub']['one'] == 1

def test_nondeep_strict_assignments_mutate():

    class Param2(Param):
        mut2 = BlobAttr(m2, deep=False)
    p = Param2()
    blob = {'mut2': {'sub': {'one': 2}}}
    p.from_blob(blob)
    assert p.mut2['sub']['one'] == 2
    p.mut2['sub']['one'] = 3
    assert p.mut2['sub']['one'] == 3
    assert blob['mut2']['sub']['one'] == 3

def test_bad_update_key():
    p = Param()
    blob = {'bad': 2}
    with pytest.raises(ValueError):
        p.from_blob(blob)

def test_bad_strict_update_key():
    p = Param()
    blob = {'mut0': 2}
    with pytest.raises(ValueError):
        p.from_blob(blob, strict=True)

def test_mutating_attribute_reference():
    p = Param()
    mut1 = p.mut1
    assert mut1['one'] == 1
    mut1['one'] = 2
    assert p.mut1['one'] == 2

def test_returned_blob_mutation():
    p = Param()
    blob = p.to_blob()
    assert blob['mut2']['sub']['one'] == 1
    blob['mut2']['sub']['one'] = 3
    assert blob['mut2']['sub']['one'] == 3
    assert p.mut2['sub']['one'] == 1