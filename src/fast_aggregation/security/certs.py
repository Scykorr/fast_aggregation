"""Самоподписанный сертификат координатора и узла.

Управляющий HTTP ходит по HTTPS. Клиент сверяет отпечаток SHA-256,
который человек видит рядом с кодом сопряжения (или в QR).
"""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def ensure_cert(directory: Path, hostnames: list[str]) -> tuple[Path, Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    cert_path = directory / "cert.pem"
    key_path = directory / "key.pem"
    if cert_path.is_file() and key_path.is_file():
        return cert_path, key_path, fingerprint_pem(cert_path.read_bytes())
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    names = [item for item in hostnames if item]
    if "localhost" not in names:
        names.append("localhost")
    san: list[x509.GeneralName] = [x509.DNSName("localhost")]
    for name in names:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(name)))
        except ValueError:
            if name != "localhost":
                san.append(x509.DNSName(name))
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, names[0])])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    pem = cert.public_bytes(serialization.Encoding.PEM)
    cert_path.write_bytes(pem)
    return cert_path, key_path, fingerprint_pem(pem)


def fingerprint_pem(pem: bytes) -> str:
    cert = x509.load_pem_x509_certificate(pem)
    digest = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    return ":".join(digest[index : index + 2] for index in range(0, len(digest), 2))


def local_names(advertise: str) -> list[str]:
    names = [advertise, "127.0.0.1"]
    try:
        names.append(socket.gethostname())
    except OSError:
        pass
    return names
