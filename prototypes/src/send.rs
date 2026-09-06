use quinn::{Connection, RecvStream, SendStream};
use tokio::io::{AsyncBufReadExt, BufReader};

pub(crate) const DEFAULT_MESSAGE: &str = "hello";
pub(crate) const MAX_MESSAGE_SIZE: usize = 64 * 1024;

pub(crate) struct BidiStream {
    send: SendStream,
    recv: RecvStream,
    recv_buf: Vec<u8>,
}

impl BidiStream {
    pub(crate) fn new(send: SendStream, recv: RecvStream) -> Self {
        Self {
            send,
            recv,
            recv_buf: Vec::new(),
        }
    }

    pub(crate) async fn write_line(&mut self, message: &str) -> Result<(), String> {
        let mut data = message.as_bytes().to_vec();
        data.push(b'\n');
        self.send
            .write_all(&data)
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub(crate) async fn read_line(&mut self) -> Result<Option<String>, String> {
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

    pub(crate) async fn finish(mut self) -> Result<(), String> {
        self.send.finish().map_err(|e| e.to_string())?;
        let _ = self.send.stopped().await;
        Ok(())
    }
}

pub async fn open_unidirectional_stream(
    connection: Connection,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut send = connection.open_uni().await?;
    send.write_all(b"test").await?;
    send.finish()?;
    let _ = send.stopped().await;
    Ok(())
}

pub async fn open_bidirectional_stream(
    connection: Connection,
) -> Result<(), Box<dyn std::error::Error>> {
    let (send, recv) = connection.open_bi().await?;
    let mut stream = BidiStream::new(send, recv);

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
