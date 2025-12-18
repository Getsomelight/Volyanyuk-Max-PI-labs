import pytest
import os
from RSA import generate_rsa_key, serialize_keys, encrypt_rsa, decrypt_rsa
from tripleDES import generate_3des_key, encrypt_3des, decrypt_3des


def test_generate_rsa_key():
    private_key, public_key = generate_rsa_key()
    assert private_key is not None
    assert public_key is not None


def test_rsa_key_types():
    private_key, public_key = generate_rsa_key()
    assert hasattr(private_key, 'public_key')
    assert hasattr(public_key, 'encrypt')


def test_rsa_encrypt_decrypt_roundtrip():
    private_key, public_key = generate_rsa_key()
    symmetric_key = b'secret-key-1234567'
    encrypted_key = encrypt_rsa(public_key, symmetric_key)
    decrypted_key = decrypt_rsa(private_key, encrypted_key)
    assert decrypted_key == symmetric_key


def test_rsa_invalid_decryption():
    private_key, _ = generate_rsa_key()
    invalid_data = b'invalid-data-too-short'
    with pytest.raises(Exception):
        decrypt_rsa(private_key, invalid_data)


def test_generate_3des_key_valid_size():
    key = generate_3des_key(128)
    assert len(key) == 16
    assert isinstance(key, bytes)


def test_generate_3des_key_invalid_size():
    with pytest.raises(ValueError, match="Invalid key length"):
        generate_3des_key(32)


@pytest.mark.parametrize("key_size", [64, 128, 192])
def test_3des_encrypt_decrypt_roundtrip(key_size):
    key = generate_3des_key(key_size)
    plaintext = "Тестовое сообщение для 3DES"
    encrypted = encrypt_3des(plaintext, key)
    decrypted = decrypt_3des(encrypted, key)
    assert decrypted == plaintext


def test_3des_invalid_key_length():
    invalid_key = b'short'
    with pytest.raises(ValueError, match="Invalid key length"):
        encrypt_3des("test", invalid_key)


def test_3des_short_encrypted_text():
    key = generate_3des_key(128)
    short_data = b'short'
    with pytest.raises(ValueError, match="Insufficient text length"):
        decrypt_3des(short_data, key)


def test_rsa_serialize_keys(tmp_path):
    private_key_path = tmp_path / "private.pem"
    public_key_path = tmp_path / "public.pem"
    private_key, public_key = generate_rsa_key()
    serialize_keys(
        private_key,
        public_key,
        str(private_key_path),
        str(public_key_path)
    )
    assert private_key_path.exists()
    assert public_key_path.exists()

