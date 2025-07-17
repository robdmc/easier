import pytest
from easier.crypt import Crypt


def test_encrypt_decrypt_round_trip():
    message = "attack at dawn"
    password = "my_password"
    crypt = Crypt()
    encrypted = crypt.encrypt(message, password)
    assert isinstance(encrypted, str)
    decrypted = crypt.decrypt(encrypted, password)
    assert decrypted == message


def test_encrypt_different_passwords():
    message = "attack at dawn"
    crypt = Crypt()
    encrypted1 = crypt.encrypt(message, "password1")
    encrypted2 = crypt.encrypt(message, "password2")
    assert encrypted1 != encrypted2


def test_decrypt_wrong_password_raises():
    message = "attack at dawn"
    password = "my_password"
    wrong_password = "wrong_password"
    crypt = Crypt()
    encrypted = crypt.encrypt(message, password)
    with pytest.raises(Exception):
        crypt.decrypt(encrypted, wrong_password)


def test_encrypt_non_ascii():
    message = "攻撃は夜明けに"
    password = "パスワード"
    crypt = Crypt()
    encrypted = crypt.encrypt(message, password)
    decrypted = crypt.decrypt(encrypted, password)
    assert decrypted == message
