# fl-compr
## TODO
1. topk: select topK weight and send using sparse
2. model performance: counting model acc / loss
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
3. simulation
train fed learning in a simulation way.
```shell
nohup python fed_sparse_simulation.py > fed_sparse.log
```
check out the client log in node-id.log file.
## Customize the clients
Referenc docs: https://flower.dev/docs/framework/tutorial-series-customize-the-client-pytorch.html