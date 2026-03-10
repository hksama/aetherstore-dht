use libp2p::request_response;
use libp2p::{Multiaddr, PeerId, SwarmBuilder, futures::StreamExt, swarm::SwarmEvent};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::time::Duration;
use tokio::sync::mpsc::channel;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RequestBody(String);
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ResponseBody(String);

enum NodeCommand {
    SendRequest { peer: PeerId, data: RequestBody },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {

    let (cmd_tx, mut cmd_rx) = channel::<NodeCommand>(32);
    let mut SwarmInfo:HashMap<PeerId,(String,Multiaddr)> = HashMap::new();

    let mut swarm = SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_quic()
        .with_behaviour(|_keypair| {
            request_response::cbor::Behaviour::<RequestBody, ResponseBody>::new(
                [(
                    libp2p::StreamProtocol::new("/aether/1.0.0"),
                    request_response::ProtocolSupport::Full,
                )],
                request_response::Config::default(),
            )
        })?
        .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(30)))
        .build();
    println!("Local PeerId: {}", swarm.local_peer_id());

    swarm.listen_on(format!("/ip4/127.0.0.1/udp/0/quic-v1").parse()?)?;

if let Some(port) = std::env::args().nth(1) {
    let peer = std::env::args().nth(2).expect("Need peer id");
    let remote: Multiaddr =
        format!("/ip4/127.0.0.1/udp/{}/quic-v1/p2p/{}", port, peer).parse()?;
    SwarmInfo.insert(peer.parse().unwrap(),(port,remote.clone()));
    swarm.dial(remote)?;

    let tx = cmd_tx.clone();

tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(3)).await;

    tx.send(NodeCommand::SendRequest {
        peer:peer.parse().unwrap(),
        data: RequestBody("hello".into()),
    }).await.unwrap();
});
}

    loop {
        tokio::select! {

            Some(cmd) = cmd_rx.recv() => {
                match cmd {
                    NodeCommand::SendRequest { peer, data } => {
                        swarm.behaviour_mut().send_request(&peer, data);
                    }
                }
            }

            event = swarm.select_next_some() => {
                match event {

                    SwarmEvent::NewListenAddr { address, .. } => {
                        println!("Listening on {address:?}");
                    }

                    SwarmEvent::Behaviour(
                        request_response::Event::Message { peer, message, .. }
                    ) => {
                        match message {

                            request_response::Message::Request { request, channel, .. } => {
                                println!("Request from {} {:?}", peer, request);

                                let result = tokio::task::spawn_blocking(move || {
                                    heavy_compute(request.0)
                                }).await.unwrap();

                                swarm.behaviour_mut()
                                    .send_response(channel, ResponseBody(result))
                                    .unwrap();
                            }

                            request_response::Message::Response { response, .. } => {
                                println!("Received Response: {:?}", response.0);
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
