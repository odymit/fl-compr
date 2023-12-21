# fl-compr

## Install Dependencies
```
# need python=3.10
pip install -r requirements.txt
```
## Start Federated Learning 
1. start server

```shell
nohup python server.py > server.log
```
check out the server log in server.log
2. start clients
```shell
# start client 1 in the first terminal
nohup python client.py --node-id 0 > node-0.log
# start client 2 in the second terminal
nohup python client.py --node-id 1 > node-1.log
```
check out the client log in node-id.log file.
## Customize the clients
Referenc docs: https://flower.dev/docs/framework/tutorial-series-customize-the-client-pytorch.html