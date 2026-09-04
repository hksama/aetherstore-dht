mod tls;

use quinn::{Connection, EndpointConfig, TokioRuntime};
use std::net::UdpSocket;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let crypto = tls::setup_quinn_crypto()?;

    //Socket Setup
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
    println!("Endpoint created");

    // Accept Incoming Connections
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

            loop {
                match connection.accept_uni().await {
                    Ok(mut recv) => match recv.read_to_end(1024 * 1024).await {
                        Ok(data) => {
                            if let Ok(text) = std::str::from_utf8(&data) {
                                println!("Received: {text:?} ({len} bytes)", len = data.len());
                            } else {
                                println!("Received {len} bytes: {data:?}", len = data.len());
                            }
                        }
                        Err(e) => eprintln!("Error reading stream: {e}"),
                    },
                    Err(e) => {
                        eprintln!("Stream accept ended: {e}");
                        break;
                    }
                }
            }
        }
    });

    // Open Connection to another node
    if let Some(request_addr) = std::env::args().nth(2) {
        // Give the accept loop time to start before we dial.
        // tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let socket = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            request_addr.parse().unwrap(),
        );
        let connection = endpoint.connect(socket, "localhost")?.await?;
        open_unidirectional_stream(connection).await?;
        println!("Sent message to {socket}");
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
    send.write_all(b"testsd9jcsd0j0sdjf0sjd9f9sd0fj9sd0f0sdf09sjd0fj9sd0jf0sdjf0sdj0f9sdj0fjsd0fjs0djf0sdjf09ssdfsd0fsd09fsd0fjsddjf0")
        .await?;
    send.finish()?;
    // Wait until the peer has acknowledged the stream end (see quinn tests).
    let _ = send.stopped().await;
    Ok(())
}
