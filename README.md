# Replication Package for “Sentiment and Uncertainty about Regulation”

**Authors:** Tara Sinclair & Zhoudan Xie  

---

## Overview

This repository provides the data and code to replicate the results from the paper titled *“Sentiment and Uncertainty about Regulation.”*

The repository includes two main directories:

1. `code`: contains Python and Stata code for text analysis, econometric analysis, and visualization.  
2. `data`: contains raw and processed data used and presented in the paper.  

Questions and comments can be directed to the authors at [tsinc@gwu.edu](mailto:tsinc@gwu.edu) or [zxie@gwu.edu](mailto:zxie@gwu.edu).  

---
 
## Data Availability and Provenance Statements

This paper analyzes the full text and metadata of newspaper articles from major U.S. newspapers, accessed through ProQuest’s TDM Studio. Due to copyright restrictions, the authors cannot distribute the full text of the news articles analyzed. However, this repository provides the ProQuest IDs and some metadata for all articles used in the analysis (available in `data/processed_data/sentiment_scores.csv`). Researchers with access to ProQuest’s content can use this information to retrieve the full text and additional metadata of the articles.  

To demonstrate the textual analysis process, this repository includes demo data for 10 news articles in XML format (located in `data/raw_data/text_sample_data`), along with the Python scripts for performing the textual analysis and estimating the sentiment and uncertainty measures. The estimated indexes using all articles are provided in the `data/processed_data` folder.  

Other economic data were obtained from publicly available data sources. See details in the **Data Source** section.  

---

## Details on Datasets and Data Sources

The `data` directory contains two folders:

- `data/raw_data`: contains data obtained from publicly available sources.
  - `data/raw_data/text_sample_data`: contains XML files for 10 sample news articles and their metadata obtained from ProQuest.
- `data/processed_data`: contains data generated from the raw data or collected manually from external sources.

**Table D1.1** lists all raw data files and sources. **Table D1.2** lists all processed data files and the name of the script that generated the data.

---

## Programs and Code

The `code` directory contains three folders:
- `code/text_analysis`: contains Python code for text analysis that estimates the regulatory sentiment and uncertainty indexes.
- `code/econometric_analysis`: contains Python code for preparing data for econometric analysis and Stata code for implementing econometric analysis.
- `code/visualization`: contains Python code for creating main and appendix figures in the paper.

---

## Computational Requirements

---

## Instructions to Replicators