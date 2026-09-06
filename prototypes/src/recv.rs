use crate::send::MAX_MESSAGE_SIZE;
use quinn::Connection;
use std::path::Path;
use tokio::io::AsyncWriteExt;

const CHUNK_SIZE: usize = 256 * 1024;
const RECV_DIR: &str = "recv";
const RECV_FILE: &str = "received.bin";

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
    let (_send, mut recv) = connection.accept_bi().await?;
    let out_path = Path::new(RECV_DIR).join(RECV_FILE);

    tokio::fs::create_dir_all(RECV_DIR).await?;
    let mut file = tokio::fs::File::create(&out_path).await?;

    let mut total = 0u64;
    while let Some(chunk) = recv.read_chunk(CHUNK_SIZE, true).await? {
        file.write_all(&chunk.bytes).await?;
        total += chunk.bytes.len() as u64;
    }

    file.sync_all().await?;
    println!("Wrote {total} bytes to {}", out_path.display());
    Ok(())
}
