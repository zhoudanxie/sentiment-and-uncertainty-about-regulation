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

- `raw_data`: contains data obtained from publicly available sources.
  - `raw_data/text_sample_data`: contains XML files for 10 sample news articles and their metadata obtained from ProQuest.
- `processed_data`: contains data generated from the raw data or collected manually from external sources.

**Table D1.1** lists all raw data files and sources. **Table D1.2** lists all processed data files and the name of the script that generated the data.

### Table D1.1: Raw Data and Sources (`data/raw_data`)

<table style="width:100%;">
  <thead>
    <tr>
      <th style="width:40%;">Data Name</th>
      <th style="width:20%;">File Name</th>
      <th style="width:40%;">Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="width:40%;">Sample news article text data</td>
      <td style="width:20%;">All XML files in <code>text_sample_data</code></td>
      <td style="width:40%;">ProQuest (2022)</td>
    </tr>
    <tr>
      <td style="width:40%;">Sample news article meta data</td>
      <td style="width:20%;"><code>text_sample_data/metadata.csv</code></td>
      <td style="width:40%;">ProQuest (2022)</td>
    </tr>
    <tr>
      <td style="width:40%;">Loughran and McDonald (LM) dictionary (2018 version)</td>
      <td style="width:20%;"><code>LoughranMcDonald_SentimentList.csv</code></td>
      <td style="width:40%;">Loughran and McDonald (2011)</td>
    </tr>
    <tr>
      <td style="width:40%;">Lexicoder Sentiment Dictionary (LSD)</td>
      <td style="width:20%;"><code>LSDsentimentWords_wStar.csv</code></td>
      <td style="width:40%;">Young and Soroka (2012)</td>
    </tr>
    <tr>
      <td style="width:40%;">Harvard General Inquirer (GI) dictionary</td>
      <td style="width:20%;"><code>GIposWords.txt</code> and <code>GInegWords.txt</code></td>
      <td style="width:40%;">Stone et al. (1966)</td>
    </tr>
    <tr>
      <td style="width:40%;">S&amp;P 500 index</td>
      <td style="width:20%;"><code>S&amp;P500.csv</code></td>
      <td style="width:40%;">S&amp;P Dow Jones Indices LLC (2022)</td>
    </tr>
    <tr>
      <td style="width:40%;">Federal funds effective rate</td>
      <td style="width:20%;"><code>FEDFUNDS.csv</code></td>
      <td style="width:40%;">Board of Governors of the Federal Reserve System (2022)</td>
    </tr>
    <tr>
      <td style="width:40%;">Employment</td>
      <td style="width:20%;"><code>PAYEMS.csv</code></td>
      <td style="width:40%;">U.S. Bureau of Labor Statistics (2022)</td>
    </tr>
    <tr>
      <td style="width:40%;">Industrial production</td>
      <td style="width:20%;"><code>INDPRO.csv</code></td>
      <td style="width:40%;">Board of Governors of the Federal Reserve System (2022)</td>
    </tr>
    <tr>
      <td style="width:40%;">Real gross domestic product</td>
      <td style="width:20%;"><code>GDPC1.csv</code></td>
      <td style="width:40%;">U.S. Bureau of Economic Analysis (2022)</td>
    </tr>
    <tr>
      <td style="width:40%;">Real gross private domestic investment</td>
      <td style="width:20%;"><code>GPDIC1.csv</code></td>
      <td style="width:40%;">U.S. Bureau of Economic Analysis (2022)</td>
    </tr>
    <tr>
      <td style="width:40%;">Economic policy uncertainty (EPU) index</td>
      <td style="width:20%;"><code>EPU_BBD.xlsx</code></td>
      <td style="width:40%;">Baker et al. (2016)</td>
    </tr>
    <tr>
      <td style="width:40%;">Categorical EPU indexes</td>
      <td style="width:20%;"><code>Categorical_EPU_Data_BBD.xlsx</code></td>
      <td style="width:40%;">Baker et al. (2016)</td>
    </tr>
    <tr>
      <td style="width:40%;">Economic sentiment index</td>
      <td style="width:20%;"><code>Shapiro_news_sentiment_data.xlsx</code></td>
      <td style="width:40%;">Shapiro et al. (2022)</td>
    </tr>
    <tr>
      <td style="width:40%;">Michigan consumer sentiment index</td>
      <td style="width:20%;"><code>MICHSENT.csv</code></td>
      <td style="width:40%;">University of Michigan (2022)</td>
    </tr>
    <tr>
      <td style="width:40%;">CBOE volatility index (VIX)</td>
      <td style="width:20%;"><code>VXO1986-2003.xls</code> and <code>VIX1990-2021.csv</code></td>
      <td style="width:40%;">Cboe Exchange, Inc. (2022)</td>
    </tr>
  </tbody>
</table>


### Table D1.2: Processed Data (`data/processed_data`)

<table style="width:100%;">
  <thead>
    <tr>
      <th style="width:40%;">Data Name</th>
      <th style="width:20%;">File Name</th>
      <th style="width:40%;">How Created</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="width:40%;">Sentiment and uncertainty scores</td>
      <td style="width:20%;"><code>sentiment_scores.csv</code></td>
      <td style="width:40%;">Estimated using sentiment analysis of all regulation-related news articles in the sample.</td>
    </tr>
    <tr>
      <td style="width:40%;">Aggregate monthly indexes of regulatory sentiment and regulatory uncertainty</td>
      <td style="width:20%;"><code>aggregate_sentiment_indexes.csv</code></td>
      <td style="width:40%;">Estimated using sentiment and uncertainty scores in <code>sentiment_scores.csv</code> (see code in <code>code/text_analysis/scripts/6_construct_aggregate_index.py</code>).</td>
    </tr>
    <tr>
      <td style="width:40%;">Categorical monthly indexes of regulatory sentiment and regulatory uncertainty</td>
      <td style="width:20%;"><code>categorical_sentiment_indexes.csv</code></td>
      <td style="width:40%;">Estimated using sentiment and uncertainty scores in <code>sentiment_scores.csv</code> (see code in <code>code/text_analysis/scripts/7_construct_categorical_index.py</code>).</td>
    </tr>
    <tr>
      <td style="width:40%;">Monthly Index of Regulatory Uncertainty</td>
      <td style="width:20%;"><code>news_attention_index.csv</code></td>
      <td style="width:40%;">Estimated using the number of regulation-related news articles in the sample (see code in <code>code/text_analysis/scripts/6_construct_aggregate_index.py</code>).</td>
    </tr>
    <tr>
      <td style="width:40%;">Aggregate quarterly indexes of regulatory sentiment and regulatory uncertainty</td>
      <td style="width:20%;"><code>aggregate_sentiment_indexes_quarterly.csv</code></td>
      <td style="width:40%;">Estimated using sentiment and uncertainty scores in <code>sentiment_scores.csv</code>.</td>
    </tr>
    <tr>
      <td style="width:40%;">Aggregate monthly indexes of regulatory sentiment and regulatory uncertainty (removing deregulation)</td>
      <td style="width:20%;"><code>aggregate_sentiment_indexes_nodereg.csv</code></td>
      <td style="width:40%;">Estimated using sentiment and uncertainty scores from sentiment analysis of regulation-related news articles excluding those mentioning deregulation.</td>
    </tr>
    <tr>
      <td style="width:40%;">Dictionary of Regulatory Noun Chunks</td>
      <td style="width:20%;"><code>dictionary_of_regulatory_noun_chunks.csv</code></td>
      <td style="width:40%;">Extracted from federal rule titles and filtered through human checking.</td>
    </tr>
    <tr>
      <td style="width:40%;">Total monthly number of news articles by newspaper</td>
      <td style="width:20%;"><code>total_article_counts_by_newspaper.xlsx</code></td>
      <td style="width:40%;">Manually collected from ProQuest.</td>
    </tr>
    <tr>
      <td style="width:40%;">Monthly number of news articles by newspaper and regulatory policy area</td>
      <td style="width:20%;"><code>article_counts_by_newspaper_and_area.csv</code></td>
      <td style="width:40%;">Calculated using identified and categorized regulation-related news articles.</td>
    </tr>
  </tbody>
</table>

---

## Programs and Code

The `code` directory contains three folders:
- `text_analysis`: contains Python code for text analysis that estimates the regulatory sentiment and uncertainty indexes.
- `econometric_analysis`: contains Python code for preparing data for econometric analysis and Stata code for implementing econometric analysis.
- `visualization`: contains Python code for creating main and appendix figures in the paper.

---

## Computational Requirements

---

## Instructions to Replicators