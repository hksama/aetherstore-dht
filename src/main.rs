use libp2p::request_response;
use libp2p::{futures::StreamExt,SwarmBuilder,Multiaddr,swarm::SwarmEvent, PeerId,};
use std::error::Error;
use std::time::Duration;
use serde::{Deserialize,Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RequestBody(String);
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ResponseBody(String);

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. Create a Swarm with QUIC
    let mut swarm = SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_quic() 
        .with_behaviour(|_keypair|{
                request_response::cbor::Behaviour::<RequestBody, ResponseBody>::new(
                [(
                    libp2p::StreamProtocol::new("/aether/1.0.0"),
                    request_response::ProtocolSupport::Full,
                )],
                request_response::Config::default(),
            )
        } )?
        .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(30)))
        .build();

    let port = std::env::args().nth(2).unwrap_or_else(|| "4000".to_string());
    swarm.listen_on(format!("/ip4/0.0.0.0/udp/{}/quic-v1", port).parse()?)?;

    if let Some(port) = std::env::args().nth(1) {    
        let remote:Multiaddr = format!("/ip4/0.0.0.0/udp/{}/quic-v1",port).parse()?;
        swarm.dial(remote)?;
        // println!("Dialed {remote}");
    }
    
    // 3. The Event Loop
loop {
        match swarm.select_next_some().await {
            SwarmEvent::NewListenAddr { address, .. } => println!("Listening on {address:?}"),
            
            // 4. Handle Incoming Events
            SwarmEvent::Behaviour(request_response::Event::Message {peer,message, connection_id }) => {
                match message {
                    request_response::Message::Request { request, channel, .. } => {
                        println!("Received Request from {peer}: {:?}", request.0);
                        let _ = swarm.behaviour_mut().send_response(channel, ResponseBody("ACK".to_string()));
                    }
                    request_response::Message::Response { response, .. } => {
                        println!("Received Response: {:?}", response.0);
                    }
                }
            }
            _ => {}
        }
    }
}
