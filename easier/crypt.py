from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from textwrap import dedent
import base64

import base64
from textwrap import dedent

class examples:
    """
    A descriptor whose only purpose is to print help text
    """

    def __get__(self, *args, **kwargs):
        print(dedent("\n                from easier import Crypt\n                message = 'attack at dawn'\n                encrypted_blob = Crypt().encrypt(message, 'my_password')\n                unecrypted_text = Crypt().decrypt(encrypted_blob, 'my_password')\n                print(f'Original: {repr(message)}')\n                print(f'Encryped Blob: {repr(encrypted_blob)}')\n                print(f'Unencrypted Message: {repr(unecrypted_text)}')\n            "))
        return None

class Crypt:
    """
    This is a utility to encrypt a string to a hex-encoded binary blob
    The code was copied from
    https://cryptography.io/en/latest/fernet/#using-passwords-with-fernet
    """
    examples = examples()

    @property
    def _kdf(self):
        return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b':h\x8a\xff\xda~}Dx\xa8\x80Q\xf3\x92\x93\x06', iterations=100000, backend=default_backend())

    def encrypt(self, message_string, password_string):
        password = password_string.encode()
        message = message_string.encode()
        key = base64.urlsafe_b64encode(self._kdf.derive(password))
        f = Fernet(key)
        token = f.encrypt(message)
        token_str = token.hex()
        return token_str

    def decrypt(self, encrypted_hex_string, password_string):
        password = password_string.encode()
        key = base64.urlsafe_b64encode(self._kdf.derive(password))
        f = Fernet(key)
        token = bytes.fromhex(encrypted_hex_string)
        message = f.decrypt(token).decode()
        return message