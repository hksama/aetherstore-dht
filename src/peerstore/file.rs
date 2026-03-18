use libp2p::{Multiaddr, PeerId};

pub struct PeerEntry {
    peer_id: PeerId,
    port: u16,
}

use std::fs::File;
use std::io::{Read, BufReader};
use std::error::Error;

fn read_peer_file(path: &str) -> Result<Vec<PeerEntry>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8;4];
    reader.read_exact(&mut magic)?;

    if &magic != b"AETH" {
        return Err("Invalid file format".into());
    }

    let mut version = [0u8;1];
    reader.read_exact(&mut version)?;

    let mut count_bytes = [0u8;4];
    reader.read_exact(&mut count_bytes)?;
    let peer_count = u32::from_be_bytes(count_bytes);

    let mut peers = Vec::with_capacity(peer_count as usize);

    for _ in 0..peer_count {

        let mut len = [0u8;1];
        reader.read_exact(&mut len)?;
        let peer_len = len[0] as usize;

        let mut peer_buf = vec![0u8;peer_len];
        reader.read_exact(&mut peer_buf)?;

        let peer_id = PeerId::from_bytes(&peer_buf)?;

        let mut port_buf = [0u8;2];
        reader.read_exact(&mut port_buf)?;
        let port = u16::from_be_bytes(port_buf);

        let mut addr_len = [0u8;1];
        reader.read_exact(&mut addr_len)?;

        let mut addr_buf = vec![0u8;addr_len[0] as usize];
        reader.read_exact(&mut addr_buf)?;

        let addr: Multiaddr = String::from_utf8(addr_buf)?.parse()?;

        peers.push(PeerEntry {
            peer_id,
            port,
        });
    }

    Ok(peers)
}