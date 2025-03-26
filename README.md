# karst_networks_gen_public

This repository is published under the BSD-3-Clause license, it accompanies the following work :
- Lauzon, D., Straubhaar, J., Renard, P. (2025). A Deep Generative Model for the Simulation of Discrete Karst Networks. *Earth and Space Science (submitted)*.

## Specific classes and functions
The directory `utils` contains python scripts defining classes and functions required to run the tests.

## Sakany cave study
The directory `Sakany` contains the materials for the Sakany cave study.

### Input data
The directory `Sakany/data` contains the input data: a pickle file with the main graph of the Sakany cave.

### Notebooks
The proposed approach is applied by running the notebooks (in `Sakany/` directory) in the following order :

- `00_graphData_collection.ipynb` : generate a collection of subgraphs (from the main graph) for data set and test set; *each (sub)graph has "centralized" position (mean of coordinates position is zero)* ; outputs are generated in a (new) directory `data_gen`.
- `01_graphRNN_model_train.ipynb` : define and train the GraphRNN model ; outputs are generated in a (new) directory `out_graphRNN_model`.
- `02_graphRNN_model_play.ipynb` (optional step) : play / test the Graph RNN model for graph generation (topology only).
- `03_graphDDPM_model_train.ipynb` : define and train the GraphDDPM model for node features generation ; outputs are generated in a (new) directory `out_graphDDPM_model`.
- `04_graphDDPM_model_play.ipynb` (optional step) : play / test the Graph DDPM model for node features generation.
- `05_gen_graph.ipynb` : generate an ensemble of graphs (topology + node features) : first, the topology is generated using the Graph RNN model, then the node features are generated using Graph DDPM model ; outputs are generated in a (new) directory `out_gen_graph`.
- `06_gen_graph_anim.ipynb` (optional step) : generate an animation of the denoising process.
- `07_gen_graph_stats.ipynb` : compute statistics on generated graphs and on the graphs from the data set.

At each step, some figures are produced and saved in a (new) directory `fig`.

## Requirements
The main required packages are `matplotlib`, `numpy`, `scipy`, `networkx`, `pyvista` (for 3D visualization), `karstnet` (for some statistics on graphs), and `torch` and `torch_geometric` (for handling neural networks). The `torch` and `torch_geometric` packages may be installed with `cuda` enabled, allowing to do computations on a Graphics Processing Unit (GPU). Versions of the packages to be installed depend on the version of the GPU.

## Acknowledgments
The authors acknowledge funding by the European Union (ERC, KARST, 101071836). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

The authors also thank the Sakany cave explorers and all those that contributed to the related dataset and that made it freely available (https://ghtopo.blog4ever.com/reseau-sakany).

The authors would also like to thank Celia Trunz for processing the raw data used in this work. 
