mod tls;
mod transport;

use quinn::{Connection, EndpointConfig, RecvStream, SendStream, TokioRuntime};
use std::net::UdpSocket;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};

const DEFAULT_MESSAGE: &str = "hello";
const MAX_MESSAGE_SIZE: usize = 64 * 1024;

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
                accept_bidirectional_streams(connection).await;
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
        open_bidirectional_stream(connection).await?;
    } else {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
        }
    }

    Ok(())
}

struct BidiStream {
    send: SendStream,
    recv: RecvStream,
    recv_buf: Vec<u8>,
}

impl BidiStream {
    async fn write_line(&mut self, message: &str) -> Result<(), String> {
        let mut data = message.as_bytes().to_vec();
        data.push(b'\n');
        self.send
            .write_all(&data)
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn read_line(&mut self) -> Result<Option<String>, String> {
        loop {
            if let Some(pos) = self.recv_buf.iter().position(|&b| b == b'\n') {
                let line = String::from_utf8_lossy(&self.recv_buf[..pos]).to_string();
                self.recv_buf.drain(..pos + 1);
                return Ok(Some(line));
            }

            if self.recv_buf.len() > MAX_MESSAGE_SIZE {
                return Err("message exceeds max size".to_string());
            }

            let mut chunk = [0u8; 256];
            match self.recv.read(&mut chunk).await.map_err(|e| e.to_string())? {
                Some(0) => return Ok(None),
                Some(n) => self.recv_buf.extend_from_slice(&chunk[..n]),
                None => {
                    if self.recv_buf.is_empty() {
                        return Ok(None);
                    }
                    let line = String::from_utf8_lossy(&self.recv_buf).to_string();
                    self.recv_buf.clear();
                    return Ok(Some(line));
                }
            }
        }
    }

    async fn finish(mut self) -> Result<(), String> {
        self.send.finish().map_err(|e| e.to_string())?;
        let _ = self.send.stopped().await;
        Ok(())
    }
}

async fn accept_unidirectional_streams(connection: Connection) {
    loop {
        match connection.accept_uni().await {
            Ok(mut recv) => match recv.read_to_end(MAX_MESSAGE_SIZE).await {
                Ok(data) => {
                    if let Ok(text) = std::str::from_utf8(&data) {
                        println!("Received (uni): {text:?} ({len} bytes)", len = data.len());
                    } else {
                        println!("Received (uni) {len} bytes: {data:?}", len = data.len());
                    }
                }
                Err(e) => eprintln!("Error reading uni stream: {e}"),
            },
            Err(e) => {
                eprintln!("Uni stream accept ended: {e}");
                break;
            }
        }
    }
}

async fn accept_bidirectional_streams(connection: Connection) {
    let (send, recv) = match connection.accept_bi().await {
        Ok(streams) => streams,
        Err(e) => {
            eprintln!("Bi stream accept ended: {e}");
            return;
        }
    };

    let mut stream = BidiStream {
        send,
        recv,
        recv_buf: Vec::new(),
    };

    while let Ok(Some(line)) = stream.read_line().await {
        println!("Received (bi): {line:?}");
        let reply = format!("echo: {line}");
        if stream.write_line(&reply).await.is_err() {
            break;
        }
    }
}

async fn open_unidirectional_stream(
    connection: Connection,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut send = connection.open_uni().await?;
    send.write_all(b"test").await?;
    send.finish()?;
    let _ = send.stopped().await;
    Ok(())
}

async fn open_bidirectional_stream(
    connection: Connection,
) -> Result<(), Box<dyn std::error::Error>> {
    let (send, recv) = connection.open_bi().await?;
    let mut stream = BidiStream {
        send,
        recv,
        recv_buf: Vec::new(),
    };

    stream.write_line(DEFAULT_MESSAGE).await?;
    let reply = stream.read_line().await?;
    println!("Reply: {:?}", reply.as_deref().unwrap_or("(closed)"));
    println!("Sent default message: {DEFAULT_MESSAGE:?}");

    println!("Enter text and press Enter to send (Ctrl+C to quit):");
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    while let Some(line) = lines.next_line().await? {
        stream.write_line(&line).await?;
        let reply = stream.read_line().await?;
        println!("Reply: {:?}", reply.as_deref().unwrap_or("(closed)"));
    }

    stream.finish().await?;
    Ok(())
}
