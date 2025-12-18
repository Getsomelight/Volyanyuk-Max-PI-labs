from file import save_binary_file
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key,
    load_pem_private_key,
)


def generate_rsa_key() -> tuple:
    """
    Generate private and public keys
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key = key
    public_key = key.public_key()
    return private_key, public_key


def serialize_keys(
    private_key,
    public_key,
    private_key_path: str,
    public_key_path: str,
) -> None:
    """
    Create serialise private and public keys and save them
    """
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    save_binary_file(private_key_path, private_pem)
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    save_binary_file(public_key_path, public_pem)


def encrypt_rsa(public_key, key) -> bytes:
    """
    Encrypt symmetric key by using public key
    """
    encrypted_key = public_key.encrypt(
        key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return encrypted_key


def decrypt_rsa(private_key, key) -> bytes:
    """
    Decrypt symmetric key by using public key
    """
    decrypted_key = private_key.decrypt(
        key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return decrypted_key
