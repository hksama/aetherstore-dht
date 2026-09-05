use libp2p::Multiaddr;
use std::env;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct NodeConfig {
    pub listen_addr: String,
    pub listen_port: u16,
    pub bootstrap_peers: Vec<Multiaddr>,
    pub peer_file: Option<String>,
    pub data_dir: String,
}

impl NodeConfig {
    pub fn from_env() -> Result<Self, Box<dyn Error>> {
        let listen_addr =
            env::var("AETHER_LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0".to_string());

        let listen_port = env::var("AETHER_LISTEN_PORT")
            .ok()
            .or_else(|| env::args().nth(1).map(String::from))
            .unwrap_or_else(|| "4001".to_string())
            .parse::<u16>()?;

        let bootstrap_peers = env::var("AETHER_BOOTSTRAP_PEERS")
            .ok()
            .map(|raw| parse_multiaddrs(&raw))
            .transpose()?
            .unwrap_or_default();

        let peer_file = env::var("AETHER_PEER_FILE").ok();

        let data_dir = env::var("AETHER_DATA_DIR").unwrap_or_else(|_| "/data".to_string());

        Ok(Self {
            listen_addr,
            listen_port,
            bootstrap_peers,
            peer_file,
            data_dir,
        })
    }

    pub fn listen_multiaddr(&self) -> Result<Multiaddr, Box<dyn Error>> {
        Ok(format!(
            "/ip4/{}/udp/{}/quic-v1",
            self.listen_addr, self.listen_port
        )
        .parse()?)
    }
}

fn parse_multiaddrs(raw: &str) -> Result<Vec<Multiaddr>, Box<dyn Error>> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<Multiaddr>().map_err(|e| e.into()))
        .collect()
}
