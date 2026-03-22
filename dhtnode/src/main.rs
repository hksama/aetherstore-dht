use libp2p::kad::{self, Behaviour as Kademlia, Event as KademliaEvent, store::MemoryStore};
use libp2p::request_response;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{Multiaddr, PeerId, SwarmBuilder, futures::StreamExt, swarm::SwarmEvent};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::time::Duration;
use tokio::sync::mpsc::channel;
mod peerstore;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RequestBody(String);
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ResponseBody(String);

#[derive(NetworkBehaviour)]
struct AetherNetworkBehaviour {
    request_response: request_response::cbor::Behaviour<RequestBody, ResponseBody>,
    kademlia: Kademlia<MemoryStore>,
}

enum NodeCommand {
    SendRequest { peer: PeerId, data: RequestBody },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let (cmd_tx, mut cmd_rx) = channel::<NodeCommand>(32);
    let mut swarm_info: HashMap<PeerId, (String, Multiaddr)> = HashMap::new();

    let mut swarm = SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_quic()
        .with_behaviour(|keypair| {
            let peer_id = PeerId::from(keypair.public());
            let store = MemoryStore::new(peer_id);
            let mut kad = Kademlia::new(peer_id, store);

            AetherNetworkBehaviour {
                request_response:
                    request_response::cbor::Behaviour::<RequestBody, ResponseBody>::new(
                        [(
                            libp2p::StreamProtocol::new("/aether/1.0.0"),
                            request_response::ProtocolSupport::Full,
                        )],
                        request_response::Config::default(),
                    ),
                kademlia: kad,
            }
        })?
        .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(30)))
        .build();

    println!("Local PeerId: {}", swarm.local_peer_id());

    // swarm.listen_on(format!("/ip4/127.0.0.1/udp/0/quic-v1").parse()?)?;

    // first argument is port
    if let Some(port) = std::env::args().nth(1) {
        // Check if port is available and if port is numeric asw
        {
            let port: u16 = port.parse()?;
        }
        swarm.listen_on(format!("/ip4/127.0.0.1/udp/{}/quic-v1", port).parse()?)?;
        let peer = std::env::args().nth(2).expect("Need peer id");
        let remote: Multiaddr =
            format!("/ip4/127.0.0.1/udp/{}/quic-v1/p2p/{}", port, peer).parse()?;
        swarm_info.insert(peer.parse().unwrap(), (port, remote.clone()));
        swarm.dial(remote)?;

        let tx = cmd_tx.clone();

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(3)).await;

            tx.send(NodeCommand::SendRequest {
                peer: peer.parse().unwrap(),
                data: RequestBody("hello".into()),
            })
            .await
            .unwrap();
        });
    }

    loop {
        tokio::select! {

                    Some(cmd) = cmd_rx.recv() => {
                        match cmd {
                            NodeCommand::SendRequest { peer, data } => {
                                // swarm.behaviour_mut().send_request(&peer, data);
                                swarm.behaviour_mut().request_response.send_request(&peer, data);
                            }
                        }
                    }
                    //libp2p events handling block
                    event = swarm.select_next_some() => {
                        match event {

                            SwarmEvent::NewListenAddr { address, .. } => {
                                println!("Listening on {address:?}");
                            }

                            SwarmEvent::Behaviour(event) => {
                                match event {

                                    AetherNetworkBehaviourEvent::RequestResponse(event) => {
                                        match event {

                                            request_response::Event::Message { peer, message, .. } => {
                                                match message {

                                                    request_response::Message::Request { request, channel, .. } => {
                                                        println!("Request from {} {:?}", peer, request);
                                                    }

                                                    request_response::Message::Response { response, .. } => {
                                                        println!("Response: {:?}", response);
                                                    }
                                                }
                                            }

                                            _ => {}
                                        }
                }

                AetherNetworkBehaviourEvent::Kademlia(event) => {
                    println!("Kad event: {:?}", event);
                }
            }
        }

                            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                                println!("Connected to {}", peer_id);
                            }

                            _ => {}
                        }
                    }
                }
    }
}
fn heavy_compute(data: String) -> String {
    // placeholder for reed solomon
    println!("Heavy computation will happen here!");
    format!("processed {}", data)
}
