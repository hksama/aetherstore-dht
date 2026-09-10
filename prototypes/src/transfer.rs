//! Two-stream transfer session: one control bidi, one data bidi.
//!
//! Control stream: `ACTL` magic + framed messages only (metadata, pause, resume, ack).
//! Data stream:    `ADAT` magic + raw file bytes only.
//!
//! Ordering (strict):
//!  1. Client opens control bidi, writes `ACTL` + Data frame
//!  2. Server accepts control bidi, reads `ACTL` + Data frame, sends Ack
//!  3. Client opens data bidi, writes `ADAT` + file chunks
//!  4. Server accepts data bidi, reads `ADAT`, then file chunks

use crate::control::{
    ControlError, ControlMessage, ControlReader, ControlWriter, MessageType,
};
use quinn::{Connection, RecvStream, SendStream};
use std::path::Path;
use tokio::io::AsyncWriteExt;

pub const CONTROL_MAGIC: &[u8; 4] = b"ACTL";
pub const DATA_MAGIC: &[u8; 4] = b"ADAT";
pub const CHUNK_SIZE: usize = 256 * 1024;

pub struct ControlSession {
    pub writer: ControlWriter,
    pub reader: ControlReader,
}

pub struct DataSender {
    send: SendStream,
}

pub struct DataReceiver {
    recv: RecvStream,
}

fn wrong_magic(expected: &'static str, got: [u8; 4], stage: &'static str) -> ControlError {
    ControlError::WrongStreamMagic {
        expected,
        got,
        stage,
    }
}

async fn write_exact(send: &mut SendStream, bytes: &[u8]) -> Result<(), ControlError> {
    send.write_all(bytes).await.map_err(ControlError::Write)?;
    Ok(())
}

async fn read_exact(recv: &mut RecvStream, buf: &mut [u8]) -> Result<(), ControlError> {
    let mut offset = 0;
    while offset < buf.len() {
        match recv.read(&mut buf[offset..]).await {
            Ok(Some(0)) => return Err(ControlError::StreamClosed),
            Ok(Some(n)) => offset += n,
            Ok(None) => return Err(ControlError::StreamClosed),
            Err(e) => return Err(ControlError::Read(e)),
        }
    }
    Ok(())
}

async fn read_magic(
    recv: &mut RecvStream,
    expected: &'static [u8; 4],
    stage: &'static str,
) -> Result<(), ControlError> {
    let mut magic = [0u8; 4];
    read_exact(recv, &mut magic).await?;
    if &magic != expected {
        return Err(wrong_magic(
            std::str::from_utf8(expected).unwrap(),
            magic,
            stage,
        ));
    }
    Ok(())
}

/// Client: open the control stream, announce transfer metadata, wait for server ready.
pub async fn client_open_control(
    connection: &Connection,
    file_size: u64,
) -> Result<ControlSession, ControlError> {
    let (mut send, recv) = connection.open_bi().await.map_err(ControlError::OpenBi)?;

    // Write magic + metadata frame in one go so the server always sees ACTL first.
    let frame = ControlMessage::data(file_size).encode()?;
    let mut opener = Vec::with_capacity(CONTROL_MAGIC.len() + frame.len());
    opener.extend_from_slice(CONTROL_MAGIC);
    opener.extend_from_slice(&frame);
    write_exact(&mut send, &opener).await?;

    let mut writer = ControlWriter::new(send);
    let mut reader = ControlReader::new(recv);
    let ready = reader
        .read_message()
        .await?
        .ok_or(ControlError::StreamClosed)?;
    if ready.msg_type != MessageType::Ack {
        return Err(ControlError::InvalidPayload(ready.msg_type));
    }
    let _ = ready.offset()?;

    Ok(ControlSession { writer, reader })
}

/// Server: accept the control stream, read metadata, acknowledge ready.
pub async fn server_accept_control(
    connection: &Connection,
) -> Result<(ControlSession, u64), ControlError> {
    let (send, mut recv) = connection.accept_bi().await.map_err(ControlError::OpenBi)?;

    read_magic(&mut recv, CONTROL_MAGIC, "control stream").await?;

    let mut reader = ControlReader::new(recv);
    let meta = reader
        .read_message()
        .await?
        .ok_or(ControlError::StreamClosed)?;
    if meta.msg_type != MessageType::Data {
        return Err(ControlError::InvalidPayload(meta.msg_type));
    }
    let file_size = meta.file_size()?;

    let mut writer = ControlWriter::new(send);
    writer.write_message(&ControlMessage::ack(0)).await?;

    Ok((ControlSession { writer, reader }, file_size))
}

/// Client: open the data stream and write the stream marker.
pub async fn client_open_data(connection: &Connection) -> Result<DataSender, ControlError> {
    let (mut send, _recv) = connection.open_bi().await.map_err(ControlError::OpenBi)?;
    write_exact(&mut send, DATA_MAGIC).await?;
    Ok(DataSender { send })
}

/// Server: accept the data stream and verify the stream marker.
pub async fn server_accept_data(connection: &Connection) -> Result<DataReceiver, ControlError> {
    let (_send, mut recv) = connection.accept_bi().await.map_err(ControlError::OpenBi)?;
    read_magic(&mut recv, DATA_MAGIC, "data stream").await?;
    Ok(DataReceiver { recv })
}

impl DataSender {
    pub async fn write_chunk(&mut self, data: &[u8]) -> Result<usize, quinn::WriteError> {
        self.send.write(data).await
    }

    pub async fn finish(mut self) -> Result<(), quinn::WriteError> {
        self.send.finish()?;
        let _ = self.send.stopped().await;
        Ok(())
    }
}

impl DataReceiver {
    pub async fn read_chunk(
        &mut self,
    ) -> Result<Option<quinn::Chunk>, quinn::ReadError> {
        self.recv.read_chunk(CHUNK_SIZE, true).await
    }
}

impl ControlSession {
    pub fn split(self) -> (ControlWriter, ControlReader) {
        (self.writer, self.reader)
    }

    pub async fn write_message(&mut self, message: &ControlMessage) -> Result<(), ControlError> {
        self.writer.write_message(message).await
    }

    pub async fn read_message(&mut self) -> Result<Option<ControlMessage>, ControlError> {
        self.reader.read_message().await
    }

    pub async fn finish(mut self) -> Result<(), ControlError> {
        self.writer.finish().await
    }
}

pub async fn write_received_file(
    recv: &mut DataReceiver,
    path: &Path,
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut file = tokio::fs::File::create(path).await?;
    let mut total = 0u64;

    while let Some(chunk) = recv.read_chunk().await? {
        file.write_all(&chunk.bytes).await?;
        total += chunk.bytes.len() as u64;
    }

    file.sync_all().await?;
    Ok(total)
}
