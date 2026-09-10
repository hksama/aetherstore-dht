//! Framed control-plane messages on a dedicated bidi stream.
//!
//! Quinn applies `stream_receive_window` per connection, not per individual stream.
//! Abuse resistance here is application-layer: capped payload size (256 B), capped
//! receive buffer (261 B), strict payload shapes per `MessageType`, and hard errors
//! on unknown types or oversize frames.
use quinn::{RecvStream, SendStream};
use std::convert::TryFrom;

pub const HEADER_SIZE: usize = 5;
pub const MAX_CONTROL_PAYLOAD: u32 = 256;
pub const MAX_CONTROL_RECV_BUF: usize = HEADER_SIZE + MAX_CONTROL_PAYLOAD as usize;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Data = 1,
    Pause = 2,
    Resume = 3,
    Ack = 4,
}

impl TryFrom<u8> for MessageType {
    type Error = ControlError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Data),
            2 => Ok(Self::Pause),
            3 => Ok(Self::Resume),
            4 => Ok(Self::Ack),
            _ => Err(ControlError::UnknownMessageType(value)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ControlMessage {
    pub msg_type: MessageType,
    pub payload: Vec<u8>,
}

#[derive(Debug)]
pub enum ControlError {
    UnknownMessageType(u8),
    PayloadTooLarge,
    BufferOverflow,
    WrongStreamMagic { expected: &'static str, got: [u8; 4], stage: &'static str },
    InvalidPayload(MessageType),
    Read(quinn::ReadError),
    Write(quinn::WriteError),
    Finish(quinn::ClosedStream),
    StreamClosed,
    OpenBi(quinn::ConnectionError),
}

impl std::fmt::Display for ControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownMessageType(v) => write!(f, "unknown control message type: {v}"),
            Self::PayloadTooLarge => write!(f, "control payload exceeds {MAX_CONTROL_PAYLOAD} bytes"),
            Self::BufferOverflow => write!(f, "control receive buffer overflow"),
            Self::WrongStreamMagic { expected, got, stage } => {
                write!(
                    f,
                    "unexpected stream magic on {stage}: expected {expected}, got {:?} ({:?})",
                    String::from_utf8_lossy(got),
                    got
                )
            }
            Self::InvalidPayload(t) => write!(f, "invalid payload for {t:?}"),
            Self::Read(e) => write!(f, "control read error: {e}"),
            Self::Write(e) => write!(f, "control write error: {e}"),
            Self::Finish(e) => write!(f, "control stream finish error: {e}"),
            Self::StreamClosed => write!(f, "control stream closed"),
            Self::OpenBi(e) => write!(f, "failed to open bidi stream: {e}"),
        }
    }
}

impl std::error::Error for ControlError {}

impl ControlMessage {
    pub fn data(file_size: u64) -> Self {
        Self {
            msg_type: MessageType::Data,
            payload: file_size.to_le_bytes().to_vec(),
        }
    }

    pub fn pause() -> Self {
        Self {
            msg_type: MessageType::Pause,
            payload: Vec::new(),
        }
    }

    pub fn resume(offset: u64) -> Self {
        Self {
            msg_type: MessageType::Resume,
            payload: offset.to_le_bytes().to_vec(),
        }
    }

    pub fn ack(offset: u64) -> Self {
        Self {
            msg_type: MessageType::Ack,
            payload: offset.to_le_bytes().to_vec(),
        }
    }

    pub fn file_size(&self) -> Result<u64, ControlError> {
        if self.msg_type != MessageType::Data || self.payload.len() != 8 {
            return Err(ControlError::InvalidPayload(self.msg_type));
        }
        Ok(u64::from_le_bytes(self.payload.as_slice().try_into().unwrap()))
    }

    pub fn offset(&self) -> Result<u64, ControlError> {
        match self.msg_type {
            MessageType::Resume | MessageType::Ack => {
                if self.payload.len() != 8 {
                    return Err(ControlError::InvalidPayload(self.msg_type));
                }
                Ok(u64::from_le_bytes(self.payload.as_slice().try_into().unwrap()))
            }
            _ => Err(ControlError::InvalidPayload(self.msg_type)),
        }
    }

    fn validate(&self) -> Result<(), ControlError> {
        if self.payload.len() as u32 > MAX_CONTROL_PAYLOAD {
            return Err(ControlError::PayloadTooLarge);
        }

        match self.msg_type {
            MessageType::Data => {
                if self.payload.len() != 8 {
                    return Err(ControlError::InvalidPayload(self.msg_type));
                }
            }
            MessageType::Pause => {
                if !self.payload.is_empty() {
                    return Err(ControlError::InvalidPayload(self.msg_type));
                }
            }
            MessageType::Resume | MessageType::Ack => {
                if self.payload.len() != 8 {
                    return Err(ControlError::InvalidPayload(self.msg_type));
                }
            }
        }

        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, ControlError> {
        self.validate()?;
        let mut frame = Vec::with_capacity(HEADER_SIZE + self.payload.len());
        frame.push(self.msg_type as u8);
        frame.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&self.payload);
        Ok(frame)
    }
}

pub struct ControlStream {
    send: SendStream,
    recv: RecvStream,
    recv_buf: Vec<u8>,
}

pub struct ControlReader {
    recv: RecvStream,
    recv_buf: Vec<u8>,
}

impl ControlReader {
    pub fn new(recv: RecvStream) -> Self {
        Self {
            recv,
            recv_buf: Vec::with_capacity(HEADER_SIZE),
        }
    }

    pub async fn read_message(&mut self) -> Result<Option<ControlMessage>, ControlError> {
        read_message_from(&mut self.recv, &mut self.recv_buf).await
    }
}

impl ControlStream {
    pub fn new(send: SendStream, recv: RecvStream) -> Self {
        Self {
            send,
            recv,
            recv_buf: Vec::with_capacity(HEADER_SIZE),
        }
    }

    pub async fn write_message(&mut self, message: &ControlMessage) -> Result<(), ControlError> {
        let frame = message.encode()?;
        self.send
            .write_all(&frame)
            .await
            .map_err(ControlError::Write)?;
        Ok(())
    }

    pub async fn read_message(&mut self) -> Result<Option<ControlMessage>, ControlError> {
        read_message_from(&mut self.recv, &mut self.recv_buf).await
    }

    pub fn split(self) -> (ControlWriter, ControlReader) {
        (
            ControlWriter::new(self.send),
            ControlReader {
                recv: self.recv,
                recv_buf: self.recv_buf,
            },
        )
    }

    fn try_decode_frame(recv_buf: &mut Vec<u8>) -> Result<Option<ControlMessage>, ControlError> {
        if recv_buf.len() < HEADER_SIZE {
            return Ok(None);
        }

        let msg_len = u32::from_le_bytes(recv_buf[1..HEADER_SIZE].try_into().unwrap());
        if msg_len > MAX_CONTROL_PAYLOAD {
            return Err(ControlError::PayloadTooLarge);
        }

        let frame_len = HEADER_SIZE + msg_len as usize;
        if recv_buf.len() < frame_len {
            return Ok(None);
        }

        let msg_type = MessageType::try_from(recv_buf[0])?;
        let payload = recv_buf[HEADER_SIZE..frame_len].to_vec();
        recv_buf.drain(..frame_len);

        let message = ControlMessage { msg_type, payload };
        message.validate()?;
        Ok(Some(message))
    }

    pub async fn finish(mut self) -> Result<(), ControlError> {
        self.send.finish().map_err(ControlError::Finish)?;
        let _ = self.send.stopped().await;
        Ok(())
    }
}

pub struct ControlWriter {
    send: SendStream,
}

impl ControlWriter {
    pub fn new(send: SendStream) -> Self {
        Self { send }
    }

    pub async fn write_message(&mut self, message: &ControlMessage) -> Result<(), ControlError> {
        let frame = message.encode()?;
        self.send
            .write_all(&frame)
            .await
            .map_err(ControlError::Write)?;
        Ok(())
    }

    pub async fn finish(mut self) -> Result<(), ControlError> {
        self.send.finish().map_err(ControlError::Finish)?;
        let _ = self.send.stopped().await;
        Ok(())
    }
}

async fn read_message_from(
    recv: &mut RecvStream,
    recv_buf: &mut Vec<u8>,
) -> Result<Option<ControlMessage>, ControlError> {
    loop {
        if let Some(message) = ControlStream::try_decode_frame(recv_buf)? {
            return Ok(Some(message));
        }

        if recv_buf.len() > MAX_CONTROL_RECV_BUF {
            return Err(ControlError::BufferOverflow);
        }

        let mut chunk = [0u8; 64];
        match recv.read(&mut chunk).await {
            Ok(Some(0)) => return Ok(None),
            Ok(Some(n)) => recv_buf.extend_from_slice(&chunk[..n]),
            Ok(None) => {
                if recv_buf.is_empty() {
                    return Ok(None);
                }
                return Err(ControlError::StreamClosed);
            }
            Err(e) => return Err(ControlError::Read(e)),
        }
    }
}
