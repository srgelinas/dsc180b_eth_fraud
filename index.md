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

# Results

# Discussion
