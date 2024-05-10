Code and data for paper: Design a Win-Win Strategy That Is Fair to Both Service Providers and Tasks
When Rejection is Not an Option.

(Accepted for publication at IJCAI 2024)

Code Requirements:
Matlab: https://www.mathworks.com/products/matlab.html
Matlab for python: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
Optimization Package for Matlab
NetworkX: https://networkx.org/documentation/stable/index.html

Main class: main.py

Example output:
Parameters: kappa 1 load 100000 arrival dist {1: 0.28935185185185186, 2: 0.28935185185185186, 3: 0.28935185185185186, 4: 0.28935185185185186} simulation duration 604800 seconds
OPT(PT) Maximum task waiting time: 1.8369723750863707 Maximum worker utilization: 0.6475113590322902
OPT(PS) Maximum task waiting time: 1.8369698150863727 Maximum worker utilization: 0.6475112302280328
SIM(PT) Maximum task waiting time: 1.8406740708689722 Maximum worker utilization: 0.6524187408353037
SIM(PS) Maximum task waiting time: 1.8451004258901673 Maximum worker utilization: 0.6525933560203617
SIM-F(PT) Maximum task waiting time: 0.5399895696376183 Maximum worker utilization: 0.6751874477345284
SIM-F(PS) Maximum task waiting time: 0.5572475377260152 Maximum worker utilization: 0.680022592039862
GTW Maximum task waiting time: 0.7220413681979566 Maximum worker utilization: 0.9598971753649823
GWU Maximum task waiting time: 1.8451941822377047 Maximum worker utilization: 0.6468060037365714

Setting parameters can be performed by changing line 89.
Graph definition is available in the file: graph_conf
Runtime: about 2 minutes with default configurations.

Task duration method: Use average time for each worker. If kappa>1, choose duration accordingly.
