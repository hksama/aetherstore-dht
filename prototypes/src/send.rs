use crate::control::{ControlError, ControlReader, MessageType};
use crate::transfer::{client_open_control, client_open_data, CHUNK_SIZE};
use quinn::{Connection, RecvStream, SendStream};
use std::io::SeekFrom;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, BufReader};
use tokio::sync::Notify;

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

/// Reads pause/resume/ack frames from the **control stream only**.
async fn run_control_reader(
    mut reader: ControlReader,
    paused: Arc<AtomicBool>,
    resume_offset: Arc<Mutex<u64>>,
    resume_notify: Arc<Notify>,
) -> Result<(), ControlError> {
    while let Some(msg) = reader.read_message().await? {
        match msg.msg_type {
            MessageType::Pause => {
                println!("Control stream: paused by peer");
                paused.store(true, Ordering::Relaxed);
            }
            MessageType::Resume => {
                let offset = msg.offset()?;
                println!("Control stream: resume at {offset}");
                *resume_offset.lock().unwrap() = offset;
                paused.store(false, Ordering::Relaxed);
                resume_notify.notify_waiters();
            }
            MessageType::Ack => {
                let offset = msg.offset()?;
                println!("Control stream: ack at {offset}");
            }
            MessageType::Data => return Err(ControlError::InvalidPayload(MessageType::Data)),
        }
    }
    Ok(())
}

pub async fn send_file(
    connection: Connection,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_size = tokio::fs::metadata(path).await?.len();

    // --- Step 1: control stream handshake (metadata only, no file bytes) ---
    let control = client_open_control(&connection, file_size).await?;
    println!("Control stream: sent metadata ({file_size} bytes), server ready");

    let paused = Arc::new(AtomicBool::new(false));
    let resume_offset = Arc::new(Mutex::new(0u64));
    let resume_notify = Arc::new(Notify::new());

    let (mut control_tx, control_rx) = control.split();
    let control_task = tokio::spawn(run_control_reader(
        control_rx,
        paused.clone(),
        resume_offset.clone(),
        resume_notify.clone(),
    ));

    // --- Step 2: data stream (raw file bytes only, opened after control ready) ---
    let mut data = client_open_data(&connection).await?;
    println!("Data stream: opened");

    let mut file = tokio::fs::File::open(path).await?;
    let mut buffer = vec![0u8; CHUNK_SIZE];
    let mut sent = 0u64;

    while sent < file_size {
        while paused.load(Ordering::Relaxed) {
            resume_notify.notified().await;
            sent = *resume_offset.lock().unwrap();
            file.seek(SeekFrom::Start(sent)).await?;
            println!("Data stream: resuming send at {sent}");
        }

        let n = file.read(&mut buffer).await?;
        if n == 0 {
            break;
        }

        let mut chunk_offset = 0;
        while chunk_offset < n {
            if paused.load(Ordering::Relaxed) {
                sent = *resume_offset.lock().unwrap();
                file.seek(SeekFrom::Start(sent)).await?;
                break;
            }
            chunk_offset += data.write_chunk(&buffer[chunk_offset..n]).await?;
        }

        if !paused.load(Ordering::Relaxed) {
            sent += n as u64;
            println!("Data stream: sent {sent}/{file_size} bytes");
        }
    }

    data.finish().await?;
    control_tx.finish().await?;

    match control_task.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e.into()),
        Err(e) => return Err(e.into()),
    }

    if sent != file_size {
        return Err(format!("incomplete send: {sent}/{file_size}").into());
    }

    println!("Sent {sent} bytes from {}", path.display());
    Ok(())
}
