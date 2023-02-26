![Image](images/graph_schema.png)
<p align="center"><em>Graph Schema</em></p>

# Motivation

Approximately 1 million Ethereum transactions occur daily, and investing more resources into Ethereum fraud detection is crucial for public confidence in blockchain and cryptocurrency adoption. As the popularity of Ethereum and other cryptocurrencies continues to grow, so does the potential for fraudulent activities within the transaction network. These activities can have a significant impact on both individual investors and the broader financial system. Additionally, the anonymous, decentralized nature of cryptocurriences makes it more challenging to identify and prevent fraudulent transactions. This highlights the need for more sophisticated and innovative fraud detection techniques. Finally, by investing in fraud detection, we can create a safer and more reliable environment for investors, which could increase overall confidence in the cryptocurrency market. Ultimately, investing in Ethereum fraud detection is a necessary step towards ensuring the integrity and sustainability of the cryptocurrency ecosystem.

This project aims to compare graph-based to non-graph based algorithms for fraud detection in Ethereum transaction networks. We will predict whether a given Ethereum wallet in the transaction graph is fraudulent or non-fraudulent, given the wallet's transaction history in the network. Graph exploration, analysis, and model building will be conducted using [TigerGraph](https://tgcloud.io/), an enterprise-scale graph data platform for advanced analytics and machine learning.


# Data

The dataset contains transaction records of 445 phishing accounts and 445 non-phishing accounts of Ethereum. We obtain 445 phishing accounts labeled by [Etherscan](etherscan.io) and the same number of randomly selected unlabeled accounts as our objective nodes. The dataset can be used to conduct node classification of financial transaction networks. 

We collect the transaction records based on an assumption that for a typical money transfer flow centered on a phishing node, the previous node of the phishing node may be a victim, and the next one to three nodes may be the bridge nodes with money laundering behaviors, as figure shows. Therefore, we collect subgraphs by [K-order sampling](https://ieeexplore.ieee.org/document/8964468) with K-in = 1, K-out = 3 for each of the 890 objective nodes and then splice them into a large-scale network with 86,623 nodes and 106,083 edges.

The dataset can be downloaded from [XBlock](http://xblock.pro/#/dataset/6), one of the largest blockchain data platforms that collects current mainstream blockchain data and has the widest coverage in the academic community.
```
@article{ wu2019tedge,
  author = "Jiajing Wu and Dan Lin and Qi Yuan and Zibin Zheng",
  title = "T-EDGE: Temporal WEighted MultiDiGraph Embedding for Ethereum Transaction Network Analysis",
  journal = "arXiv preprint arXiv:1905.08038",
  year = "2019",
  URL = "https://arxiv.org/abs/1905.08038"
}
```

# Processing

# Methods

## SVM

Our group first started off with a support vector machine model (SVM) to act as our baseline. This is a supervised machine learning algorithm with the objective to find a hyperplane in an N-dimensional space (N being the # of features) that classifies the data points. Because this is a supervised learning approach, we filtered out the unlabelled data and achieved an accuracy of 48% with this classification model. 

## Node2vec

Node2vec as an algorithm solves the problem of transforming the information inherent within a network into a workable numeric representation by transforming each node into a vector. The first step in this process is done through a random walk algorithm. In our case, the edges between the nodes are given weights corresponding to the transaction amount between accounts. These weights are used to simulate random paths between nodes from the network. In the next step, the skip-gram model works with these paths to learn and creates a node embedding for each node. How this is done is that it reads the random paths taken, and learns which nodes are likely to precede another node. These embeddings then allow us to determine the makings of a fraudulent account. We achieved the best results through the pytorch implementation, with roughly a 76.6% accuracy on the test set. The pytorch package allows us to make use of the large amount of unsupervised nodes present in our dataset, improving performance.
		
## GraphSage

GraphSage is most known for its inductive nature which allows it to better adapt to previously unseen nodes. In a setting like Ethereum transaction networks where the network evolves with new nodes and edges every day, there would be a benefit to using a model that does not have to be retrained every time new information is introduced. Additionally, in comparison to methods like node2vec, GraphSage has the advantage of being able to learn from node features. GraphSage works by starting at a node and aggregating information on its neighbors to generate the embeddings. The pytorch package for this model produced an accuracy of 81.9%, a noticeable upgrade compared to node2vec. This difference can be attributed to the fact that in a largely unsupervised dataset, an inductive approach may outperform the deductive approach.

## GCN

We worked with GCNs in our first quarter on our Clickbait Detection algorithm with TextGCN and we have implemented a version for this task as well. Much of the structure is the same; we create an adjacency matrix based on the edges of the network then propagate steps forwards (3 in our case) then form node embeddings based on these. From there, we can iterate through the nodes and update the node features by hashing the features of each neighboring node. The GCN model from pytorch produced an accuracy of 79.6%
	
## Graph Attention Network

We also tested the performance of Graph Attention Networks (GAT). GATs are a type of graph neural network that leverages masked self-attention which addresses the shortcomings of prior methods based on graph convolutions or their approximations. With GATs, we are able to extract meaningful node features by implicitly speicfying different weights got different nodes in a neighborhood, without requiring costly matrix operations nor knowing the underlying graph structure upfront. GAT outperformed GCN and GraphSAGE on node classification tasks on the Cora dataset, so we hypothesized that this model may have similar performance on our dataset compared to other models. The GAT model from pytorch produced an accuracy of 78.5%, and slightly underperformed compared to GCN and GraphSAGE.

## Topology Adaptive - GCN

The final model we evaluated was Topology Adaptive GCN (TA-GCN). TA-GCN is a graph neural network defined in the vertex domain and outperforms many spectral graph convolutional neural networks (CNNs). TA-GCN utilizes a set of fixed-size learnable filters to perform convolutions on graphs. The topologies of these filters are adaptive to the graph’s topology when scanned for convolutions. TA-GCN inherits the properties of convolutions in CNNs for grid structured data, yet doesn’t require approximation to compute the convolution, leading to greater performance relative to spectral CNNs. TA-GCN is also computationally simpler than other graph neural networks. Additionally TA-GCN tends to achieve comparable or better accuracy than GCN and GraphSAGE for datasets with larger and sparser graph structures. We hypothesized that the sparsity of our transaction network will be beneficial for TA-GCNs performance in our node classification prediction task. We implemented a TA-GCN model using pytorch, which produced an accuracy of 82.2%, outperforming all of our models for comparison.


# Results

# Discussion
