import base64
from textwrap import dedent


class examples:
    """
    A descriptor whose only purpose is to print help text
    """

    def __get__(self, *args, **kwargs):
        print(
            dedent(
                """
                from easier import Crypt
                message = 'attack at dawn'
                encrypted_blob = Crypt().encrypt(message, 'my_password')
                unecrypted_text = Crypt().decrypt(encrypted_blob, 'my_password')
                print(f'Original: {repr(message)}')
                print(f'Encryped Blob: {repr(encrypted_blob)}')
                print(f'Unencrypted Message: {repr(unecrypted_text)}')
            """
            )
        )
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
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes

        return PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b":h\x8a\xff\xda~}Dx\xa8\x80Q\xf3\x92\x93\x06",
            iterations=100000,
            backend=default_backend(),
        )

    def encrypt(self, message_string, password_string):
        from cryptography.fernet import Fernet

        password = password_string.encode()
        message = message_string.encode()
        key = base64.urlsafe_b64encode(self._kdf.derive(password))
        f = Fernet(key)
        token = f.encrypt(message)
        token_str = token.hex()
        return token_str

    def decrypt(self, encrypted_hex_string, password_string):
        from cryptography.fernet import Fernet

        password = password_string.encode()
        key = base64.urlsafe_b64encode(self._kdf.derive(password))
        f = Fernet(key)
        token = bytes.fromhex(encrypted_hex_string)
        message = f.decrypt(token).decode()
        return message
