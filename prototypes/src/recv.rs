use crate::control::{ControlError, ControlMessage, ControlReader, ControlWriter, MessageType};
use crate::send::MAX_MESSAGE_SIZE;
use crate::transfer::{server_accept_control, server_accept_data};
use quinn::Connection;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::sync::mpsc;

const RECV_DIR: &str = "recv";
const RECV_FILE: &str = "received.bin";

enum TransferCommand {
    Pause,
    Resume,
}

async fn run_server_control_reader(mut reader: ControlReader) -> Result<(), ControlError> {
    while let Some(msg) = reader.read_message().await? {
        match msg.msg_type {
            MessageType::Ack => {
                let offset = msg.offset()?;
                println!("Control stream: peer ack at {offset}");
            }
            other => return Err(ControlError::InvalidPayload(other)),
        }
    }
    Ok(())
}

async fn run_server_stdin_commands(tx: mpsc::Sender<TransferCommand>) {
    let stdin = tokio::io::BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let cmd = match line.trim() {
            "p" | "P" => Some(TransferCommand::Pause),
            "r" | "R" => Some(TransferCommand::Resume),
            _ => None,
        };
        if let Some(cmd) = cmd {
            if tx.send(cmd).await.is_err() {
                break;
            }
        }
    }
}

async fn handle_pause(
    control_tx: &mut ControlWriter,
    total: u64,
) -> Result<(), ControlError> {
    control_tx.write_message(&ControlMessage::pause()).await?;
    println!("Control stream: paused receive at {total} bytes (client should stop sending)");
    Ok(())
}

async fn handle_resume(
    control_tx: &mut ControlWriter,
    total: u64,
) -> Result<(), ControlError> {
    control_tx
        .write_message(&ControlMessage::resume(total))
        .await?;
    control_tx.write_message(&ControlMessage::ack(total)).await?;
    println!("Control stream: resumed receive at {total} bytes");
    Ok(())
}

pub async fn accept_unidirectional_streams(connection: Connection) {
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

pub async fn accept_bidirectional_streams(connection: Connection) {
    use crate::send::BidiStream;

    let (send, recv) = match connection.accept_bi().await {
        Ok(streams) => streams,
        Err(e) => {
            eprintln!("Bi stream accept ended: {e}");
            return;
        }
    };

    let mut stream = BidiStream::new(send, recv);

    while let Ok(Some(line)) = stream.read_line().await {
        println!("Received (bi): {line:?}");
        let reply = format!("echo: {line}");
        if stream.write_line(&reply).await.is_err() {
            break;
        }
    }
}

pub async fn accept_file_transfer(
    connection: Connection,
) -> Result<(), Box<dyn std::error::Error>> {
    let (control, expected_size) = server_accept_control(&connection).await?;
    println!("Control stream: expecting {expected_size} bytes on data stream");
    println!("Type p + Enter to pause, r + Enter to resume receiving");

    let out_path = Path::new(RECV_DIR).join(RECV_FILE);
    tokio::fs::create_dir_all(RECV_DIR).await?;

    let (mut control_tx, control_rx) = control.split();
    let control_task = tokio::spawn(run_server_control_reader(control_rx));

    let (cmd_tx, mut cmd_rx) = mpsc::channel(8);
    tokio::spawn(run_server_stdin_commands(cmd_tx));

    let mut data = server_accept_data(&connection).await?;
    println!("Data stream: accepted");

    let mut file = tokio::fs::File::create(&out_path).await?;
    let mut total = 0u64;
    let mut paused = false;

    while total < expected_size {
        tokio::select! {
            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(TransferCommand::Pause) => {
                        if !paused {
                            handle_pause(&mut control_tx, total).await?;
                            paused = true;
                        }
                    }
                    Some(TransferCommand::Resume) => {
                        if paused {
                            handle_resume(&mut control_tx, total).await?;
                            paused = false;
                        }
                    }
                    None => {}
                }
            }
            chunk = data.read_chunk(), if !paused => {
                match chunk? {
                    Some(chunk) => {
                        file.write_all(&chunk.bytes).await?;
                        total += chunk.bytes.len() as u64;
                        println!("Data stream: received {total}/{expected_size} bytes");
                    }
                    None => {
                        return Err("data stream closed before transfer complete".into());
                    }
                }
            }
        }
    }

    file.sync_all().await?;
    control_tx.finish().await?;

    if let Err(e) = control_task.await? {
        return Err(e.into());
    }

    println!(
        "Wrote {total} bytes to {} (expected {expected_size})",
        out_path.display()
    );

    if total != expected_size {
        return Err(format!("size mismatch: got {total}, expected {expected_size}").into());
    }

    Ok(())
}
