use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use rcgen::generate_simple_self_signed;
use ring::hkdf;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::{
    ClientConfig as RustlsClientConfig, RootCertStore, ServerConfig as RustlsServerConfig,
};
use rustls_pemfile::{certs, pkcs8_private_keys};
use std::error::Error;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct TlsMaterial {
    pub cert_chain: Vec<CertificateDer<'static>>,
    pub private_key: PrivateKeyDer<'static>,
    pub trust_anchor: CertificateDer<'static>,
}

pub struct QuinnCrypto {
    pub client_config: quinn::ClientConfig,
    pub server_config: quinn::ServerConfig,
}

/// Load PEM material and build Quinn client + server TLS configs.
pub fn setup_quinn_crypto() -> Result<QuinnCrypto, Box<dyn Error>> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .unwrap();

    let tls = load_or_create_tls_material()?;

    let rustls_config = RustlsServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(tls.cert_chain, tls.private_key)?;

    // Token Key Setup
    let master_key = hkdf::Salt::new(hkdf::HKDF_SHA256, b"p10dpIwlclaocl39L%7&2#? d(1ps%b")
        .extract(b"Aetherstore Setup Secret spAzH9MNoS0pgmh28Vyz5WG3J2SRqEagIugHF8cXb7Mp1JUrNy");
    let token_key = Arc::new(master_key);
    println!("TokenKey setup complete");

    // Client Config
    let mut roots = RootCertStore::empty();
    roots.add(tls.trust_anchor)?;
    let client_crypto = RustlsClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let quic_client_config = QuicClientConfig::try_from(client_crypto)?;
    let client_config = quinn::ClientConfig::new(Arc::new(quic_client_config));
    println!("ClientConfig complete");

    // Server Config
    let quic_crypto = QuicServerConfig::try_from(rustls_config).unwrap();
    let server_config = quinn::ServerConfig::new(Arc::new(quic_crypto), token_key);

    Ok(QuinnCrypto {
        client_config,
        server_config,
    })
}

pub fn cert_dir() -> PathBuf {
    std::env::var("AETHER_CERT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("prototypes/certs"))
}

pub fn load_or_create_tls_material() -> Result<TlsMaterial, Box<dyn Error>> {
    let cert_dir = cert_dir();
    let cert_path = cert_dir.join("cert.pem");
    let key_path = cert_dir.join("key.pem");

    if cert_path.exists() && key_path.exists() {
        println!("Loading TLS material from {}", cert_dir.display());
        load_tls_material(&cert_path, &key_path)
    } else {
        println!(
            "TLS material not found in {}; generating and writing PEM files",
            cert_dir.display()
        );
        create_and_store_tls_material(&cert_dir, &cert_path, &key_path)
    }
}

fn load_tls_material(
    cert_path: &Path,
    key_path: &Path,
) -> Result<TlsMaterial, Box<dyn Error>> {
    let cert_chain = read_certs(cert_path)?;
    let private_key = read_private_key(key_path)?;
    let trust_anchor = cert_chain
        .first()
        .cloned()
        .ok_or("certificate file contains no certificates")?;

    Ok(TlsMaterial {
        cert_chain,
        private_key,
        trust_anchor,
    })
}

fn create_and_store_tls_material(
    cert_dir: &Path,
    cert_path: &Path,
    key_path: &Path,
) -> Result<TlsMaterial, Box<dyn Error>> {
    fs::create_dir_all(cert_dir)?;

    let generated = generate_simple_self_signed(vec!["localhost".into()])?;
    fs::write(cert_path, generated.cert.pem())?;
    fs::write(key_path, generated.signing_key.serialize_pem())?;
    chmod_unix(key_path, 0o600)?;
    chmod_unix(cert_path, 0o644)?;

    println!("Wrote {}", cert_path.display());
    println!("Wrote {}", key_path.display());

    load_tls_material(cert_path, key_path)
}

fn read_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>, Box<dyn Error>> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let certs = certs(&mut reader).collect::<Result<Vec<_>, _>>()?;
    if certs.is_empty() {
        return Err(format!("no certificates found in {}", path.display()).into());
    }
    Ok(certs)
}

fn read_private_key(path: &Path) -> Result<PrivateKeyDer<'static>, Box<dyn Error>> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut keys = pkcs8_private_keys(&mut reader).collect::<Result<Vec<_>, _>>()?;
    let key = keys
        .pop()
        .ok_or_else(|| format!("no private key found in {}", path.display()))?;
    Ok(PrivateKeyDer::Pkcs8(key))
}

#[cfg(unix)]
fn chmod_unix(path: &Path, mode: u32) -> Result<(), Box<dyn Error>> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(not(unix))]
fn chmod_unix(_path: &Path, _mode: u32) -> Result<(), Box<dyn Error>> {
    Ok(())
}
