import os
from Crypto.Cipher import DES3
from Crypto.Util.Padding import pad
from Crypto.Util.Padding import unpad
from Crypto.Random import get_random_bytes


def generate_3des_key(size: int) -> bytes:
    """
    Generate symmetric key by 3DEC
    """
    valid_size = {64, 128, 192}
    if size not in valid_size:
        raise ValueError("Invalid key length")
    key_bytes = size // 8
    return get_random_bytes(key_bytes)


def encrypt_3des(text: str, key: bytes) -> bytes:
    """
    Encrypt input file by using symmetric key
    """
    valid_size = {8, 16, 24}
    if len(key) not in valid_size:
        raise ValueError("Invalid key length")
    iv = os.urandom(8)
    cipher = DES3.new(key, DES3.MODE_CBC, iv)
    padded_text = pad(text.encode("utf-8"), DES3.block_size)
    return iv + cipher.encrypt(padded_text)


def decrypt_3des(encrypted_text: bytes, key: bytes) -> str:
    """
    Decrypt input file by using symmetric key
    """
    if len(encrypted_text) < 8:
        raise ValueError("Insufficient text length for IV extraction")
    iv = encrypted_text[:8]
    ciphertext = encrypted_text[8:]
    try:
        cipher = DES3.new(key, DES3.MODE_CBC, iv)
        padded_text = cipher.decrypt(ciphertext)
        return unpad(padded_text, DES3.block_size).decode("utf-8")
    except Exception:
        raise ValueError("Error")
