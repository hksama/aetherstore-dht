mod tls;

use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use quinn::{Connection, EndpointConfig, TokioRuntime};
use ring::hkdf;
use rustls::{
    ClientConfig as RustlsClientConfig, RootCertStore, ServerConfig as RustlsServerConfig,
};
use std::net::UdpSocket;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .unwrap();
    let port = std::env::args().nth(1).unwrap_or("8001".into());
    let tls = tls::load_or_create_tls_material()?;

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

    //Socket Setup
    let endpoint_config = EndpointConfig::default();
    let socket = UdpSocket::bind(format!("0.0.0.0:{port}"))?;
    let mut endpoint = quinn::Endpoint::new(
        endpoint_config,
        Some(server_config),
        socket,
        Arc::new(TokioRuntime),
    )?;
    endpoint.set_default_client_config(client_config);
    println!("Endpoint created");

    // Accept Incoming Connections
    let endpoint_incoming = endpoint.clone();

    tokio::spawn(async move {
        loop {
            if let Some(incoming) = endpoint_incoming.accept().await {
                match incoming.await {
                    Ok(connection) => {
                        println!(
                            "Incoming connection accepted from {}!",
                            connection.remote_address()
                        );
                        tokio::spawn(async move {
                            if let Err(e) = receive_unidirectional_stream(connection).await {
                                println!("Error receiving unidirectional stream: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        println!("Error receiving connection: {}", e);
                    }
                }
            }
        }
    });

    // Open Connection to another node
    if let Some(request_addr) = std::env::args().nth(2) {
        let socket = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            request_addr.parse().unwrap(),
        );
        let connection = endpoint.clone().connect(socket, "localhost")?.await?;
        open_unidirectional_stream(connection.clone()).await?;
    }
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    Ok(())
}

async fn open_unidirectional_stream(
    connection: Connection,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut send = connection.open_uni().await?;
    send.write_all(b"test").await?;
    send.finish()?;
    Ok(())
}

async fn receive_unidirectional_stream(
    connection: Connection,
) -> Result<(), Box<dyn std::error::Error>> {
    while let Ok(mut recv) = connection.accept_uni().await {
        // Because it is a unidirectional stream, we can only receive not send back.
        println!("{:?}", recv.read_to_end(50).await?);
    }
    Ok(())
}
