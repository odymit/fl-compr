# fl-compr
## TODO
1. send gradient instead of model weight
2. topk: select topK weight and send using sparse
3. model performance: counting model acc / loss
4. visualization of model weight change
## Install Dependencies
```
# need python=3.10
pip install -r requirements.txt
```
## Start Federated Learning Simulation
### supported mode
- [x] default
- [x] topk
- [ ] randomk
- [ ] quantization
### supported metrics
- [x] acc
- [x] data
- [x] time

```shell
nohup python fed_baseline.py > baseline.log
nohup python fed_baseline.py --mode topk > topk.log
```
### Results


## Customize the clients
Referenc docs: https://flower.dev/docs/framework/tutorial-series-customize-the-client-pytorch.html