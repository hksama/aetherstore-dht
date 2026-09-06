use crate::send::{BidiStream, MAX_MESSAGE_SIZE};
use quinn::Connection;

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
