mod recv;
mod send;
mod tls;
mod transport;

use quinn::{EndpointConfig, TokioRuntime};
use recv::accept_file_transfer;
use send::{open_bidirectional_stream, send_file};
use std::net::UdpSocket;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let crypto = tls::setup_quinn_crypto()?;

    let port = std::env::args().nth(1).unwrap_or("8001".into());
    let endpoint_config = EndpointConfig::default();
    let socket = UdpSocket::bind(format!("0.0.0.0:{port}"))?;
    let mut endpoint = quinn::Endpoint::new(
        endpoint_config,
        Some(crypto.server_config),
        socket,
        Arc::new(TokioRuntime),
    )?;
    endpoint.set_default_client_config(crypto.client_config);
    println!("Endpoint created on port {port}");

    let endpoint_incoming = endpoint.clone();

    tokio::spawn(async move {
        loop {
            let Some(incoming) = endpoint_incoming.accept().await else {
                break;
            };

            let connection = match incoming.await {
                Ok(connection) => connection,
                Err(e) => {
                    eprintln!("Error receiving connection: {e}");
                    continue;
                }
            };

            println!(
                "Incoming connection accepted from {}!",
                connection.remote_address()
            );

            tokio::spawn(async move {
                if let Err(e) = accept_file_transfer(connection).await {
                    eprintln!("File receive failed: {e}");
                }
            });
        }
    });

    if let Some(request_addr) = std::env::args().nth(2) {
        let remote = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            request_addr.parse().unwrap(),
        );
        let connection = endpoint.connect(remote, "localhost")?.await?;
        println!("Connected to {remote}");
        if let Some(file_path) = std::env::args().nth(3) {
            send_file(connection, Path::new(&file_path)).await?;
        } else {
            open_bidirectional_stream(connection).await?;
        }
    } else {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
        }
    }

    Ok(())
}
