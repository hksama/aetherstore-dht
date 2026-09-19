
  1. Build and start all 10 containers:
    docker compose -f docker-compose.sim.yml up -d --build
  
  2. Check running containers:
    docker compose -f docker-compose.sim.yml ps

  3. To run a single instance without docker(server only): 
    ```cargo r --bin prototypes <port>``` eg. ```cargo r --bin prototypes 8001 ``` .
    This starts an instance at port 8001

  4. To run a single instance without docker(server and client):
    ```cargo r --bin prototypes <port> <recv port> "absolute filepath"``` eg. ```cargo r --bin prototypes 8001 8000 "~/aetherstore-dht/send/100M.bin"```.
    This starts an instance at port 8001 and sends the file to port 8000.
    
 
   