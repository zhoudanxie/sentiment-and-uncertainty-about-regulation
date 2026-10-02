# Replication Package for “Sentiment and Uncertainty about Regulation”

**Authors:** Tara Sinclair & Zhoudan Xie  

**Cite:** Sinclair T.M., & Xie Z. (2026) Sentiment and uncertainty about regulation. *Macroeconomic Dynamics*. 30. [https://doi.org/10.1017/S1365100526101394](https://doi.org/10.1017/S1365100526101394).

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

**Table D1.1** lists all raw data files and sources. **Table D1.2** lists all processed data files and description of how each was created.

### Table D1.1: Raw Data (`data/raw_data`)

| Data Name                                            | File Name | Source                                                  |
|------------------------------------------------------|--------------------|---------------------------------------------------------|
| Sample news article text data                        | All XML files in `text_sample_data` | ProQuest (2022)                                         |
| Sample news article meta data                        | `text_sample_data/metadata.csv` | ProQuest (2022)                                         |
| Loughran and McDonald (LM) dictionary (2018 version) | `LoughranMcDonald_SentimentList.csv` | Loughran and McDonald (2011)                            |
| Lexicoder Sentiment Dictionary (LSD)                 | `LSDsentimentWords_wStar.csv` | Young and Soroka (2012)                                 |
| Harvard General Inquirer (GI) dictionary             | `GIposWords.txt` and `GInegWords.txt` | Stone et al. (1966)                                     |
| S&P 500 index                                        | `S&P500.csv` | S&P Dow Jones Indices LLC (2022)                        |
| Federal funds effective rate                         | `FEDFUNDS.csv` | Board of Governors of the Federal Reserve System (2022) |
| Employment                                           | `PAYEMS.csv` | U.S. Bureau of Labor Statistics (2022)                  |
| Industrial production                                | `INDPRO.csv` | Board of Governors of the Federal Reserve System (2022) |
| Real gross domestic product                          | `GDPC1.csv` | U.S. Bureau of Economic Analysis (2022)                 |
| Real gross private domestic investment               | `GPDIC1.csv` | U.S. Bureau of Economic Analysis (2022)                 |
| Economic policy uncertainty (EPU) index              | `EPU_BBD.xlsx` | Baker et al. (2016)                                     |
| Categorical EPU indexes                              | `Categorical_EPU_Data_BBD.xlsx` | Baker et al. (2016)                                     |
| Economic sentiment index                             | `Shapiro_news_sentiment_data.xlsx` | Shapiro et al. (2022)                                   |
| Michigan consumer sentiment index                    | `MICHSENT.csv` | University of Michigan (2022)                           |
| CBOE volatility index (VIX)                          | `VXO1986-2003.xls` and `VIX1990-2021.csv` | Cboe Exchange, Inc. (2022) |


### Table D1.2: Processed Data (`data/processed_data`)

| Data Name                                                                                            | File Name | How Created                                                                                                                                                                                                         |
|------------------------------------------------------------------------------------------------------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Aggregate monthly indexes of regulatory sentiment and regulatory uncertainty                         | `aggregate_sentiment_indexes.csv` | <details><summary>Details</summary>Estimated using sentiment and uncertainty scores in `sentiment_scores.csv` (see `code/text_analysis/scripts/6_construct_aggregate_index.py`).</details>                          |
| Categorical monthly indexes of regulatory sentiment and regulatory uncertainty                       | `categorical_sentiment_indexes.csv` | <details><summary>Details</summary>Estimated using sentiment and uncertainty scores in `sentiment_scores.csv` (see `code/text_analysis/scripts/7_construct_categorical_index.py`).</details>                        |
| Monthly index of regulatory news attention to regulation                                             | `news_attention_index.csv` | <details><summary>Details</summary>Estimated using the number of regulation-related news articles in the sample (see `code/text_analysis/scripts/6_construct_aggregate_index.py`).</details>                        |
| Sentiment and uncertainty scores for news articles                                                   | `sentiment_scores.csv` | <details><summary>Details</summary>Estimated using sentiment analysis of all regulation-related news articles in the sample (see `code/text_analysis/scripts/4_sentiment_analysis.py`).</details>                   |
| Aggregate quarterly indexes of regulatory sentiment and regulatory uncertainty                       | `aggregate_sentiment_indexes_quarterly.csv` | <details><summary>Details</summary>Estimated using the same approach as the monthly aggregate indexes (see `code/text_analysis/scripts/7_construct_categorical_index.py`).</details>                                |
| Aggregate monthly indexes of regulatory sentiment and regulatory uncertainty (removing deregulation) | `aggregate_sentiment_indexes_nodereg.csv` | <details><summary>Details</summary>Estimated using the same approach as the monthly aggregate indexes while excluding news articles that mention deregulation (see code in `code/text_analysis/scripts`).</details> |
| Dictionary of regulatory noun chunks                                                                 | `dictionary_of_regulatory_noun_chunks.csv` | <details><summary>Details</summary>Extracted from federal rule titles and filtered through human checking.</details>                                                                                                |
| Monthly number of all news articles by newspaper                                                     | `total_article_counts_by_newspaper.xlsx` | <details><summary>Details</summary>Manually collected from ProQuest.</details>                                                                                                                                      |
| Monthly number of regulation-related news articles by newspaper and regulatory policy area           | `article_counts_by_newspaper_and_area.csv` | <details><summary>Details</summary>Calculated using identified and categorized regulation-related news articles (see `code/text_analysis/scripts/5_categorize_articles.py`).</details>                              |

---

## Programs and Code

The `code` directory contains three folders:
- `text_analysis`: contains Python code for text analysis that estimates the regulatory sentiment and uncertainty indexes.
- `econometric_analysis`: contains Python code for preparing data for econometric analysis and Stata code for implementing econometric analysis.
- `visualization`: contains Python code for creating main and appendix figures in the paper.

**Table D2.1-3** list the scripts in each folder and their outputs.

### Table D2.1: Text Analysis Code (`code/text_analysis`)
| Script Name | Description                                                                                                               | Output                                                                                                                        |
|----------|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `text_analysis_master.py` | A master file to execute all Python scripts in `code/text_analysis/scripts`.                                              | See outputs of individual scripts below.                                                                                      |
| `1_parse_xml.py` | Parses the full text and metadata of each news article from XML files (executed on demo data).                            | `data/raw_data/text_sample_data/sample_output/parsed_xml.pkl`                                                                 |
| `2_extract_reg_sections.py` | Extracts “regulatory sections” from news articles (executed on demo data).                                                | `data/raw_data/text_sample_data/sample_output/reg_sections.pkl`                                                               |
| `3_identify_relevance.py` | Refines regulatory sections using human-checked dictionary of regulatory noun chunks (executed on demo data).             | `reg_sections.pkl` and `noun_chunk_occurrences.csv` in `data/raw_data/text_sample_data/sample_output`                         |
| `4_sentiment_analysis.py` | Calculates sentiment and uncertainty scores from regulation sections in news articles (executed on demo data).            | `data/raw_data/text_sample_data/sample_output/sentiment_scores.pkl`                                                           |
| `5_categorize_articles.py` | Categorizes regulatory-related news articles by regulatory policy area (executed on demo data).                           | `article_counts_by_newspaper_and_area.csv` and `noun_chunk_occurrences.csv` in `data/raw_data/text_sample_data/sample_output` |
| `6_construct_aggregate_index.py` | Estimates aggregate indexes of regulatory sentiment and regulatory uncertainty and index of news attention to regulation. | `aggregate_sentiment_indexes.csv` and `news_attention_index.csv` in `data/processed_data`                                     |
| `7_construct_categorical_index.py` | Estimates categorical indexes of regulatory sentiment and regulatory uncertainty.                                         | `data/processed_data/categorical_sentiment_indexes.csv`                                                                       |

### Table D2.2: Econometric Analysis Code (`code/econometric_analysis`)
| Script Name                | Description                                                                                                                                                   | Output                                                       |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| `prepare_economic_data.py` | Cleans and merges raw and processed data to create Stata data files for econometric analysis.                                                                 | All files in `data/processed_data/data_for_analysis`         |
| `econometric_master.do`    | A master file to execute all Stata code (.do files) in `code/econometric_analysis/do_files`.                                                                  | See outputs of individual scripts below.                     |
| `lp_aggregate.do`          | Runs local projections using aggregate monthly indexes of regulatory sentiment and regulatory uncertainty.                                                    | `output/[lm/gi/lsd/pc/rpu]_[lgdp/lemp].dta`                  |
| `lp_area.do`               | Runs local projections using categorical monthly indexes of regulatory sentiment and regulatory uncertainty.                                                  | `output/[lm/gi/lsd/pc/rpu]_dda[1-14]_[lgdp/lemp].dta`        |
| `lp_control.do`            | Runs local projections using aggregate regulatory indexes while controlling for other sentiment or uncertainty indexes.                                       | `output/[lm/rpu]_[lgdp/lemp]_[mich/newssent/vix/epu].dta`    |
| `lp_quarterly.do`          | Runs local projections using aggregate quarterly indexes of regulatory sentiment and regulatory uncertainty.                                                  | `output/[lm/gi/lsd/pc/rpu]_[lgdp/lemp/lgross]_quarterly.dta` |
| `lp_nodereg.do`            | Runs local projections using aggregate regulatory indexes constructed from articles that exclude deregulation.                                                | `output/[lm/rpu]_[lgdp/lemp]_nodereg.dta`                    |
| `lp_h36.do`                | Runs local projections using aggregate regulatory indexes over a 36-month horizon.                                                                            | `output/[lm/gi/lsd/pc/rpu]_[lgdp/lemp]_h36.dta`              |
| `lp_sent_interaction.do`   | Runs local projections using aggregate regulatory uncertainty index while adding an interaction term between regulatory sentiment and regulatory uncertainty. | `output/lm_[lgdp/lemp]_interaction.dta`                      |
| `lp_rpu_interaction.do`    | Runs local projections using aggregate regulatory uncertainty index while adding an interaction term between regulatory sentiment and regulatory uncertainty. | `output/rpu_[lgdp/lemp]_interaction.dta`                     |
| `var_lm.do`                | Runs VAR using aggregate regulatory sentiment index.                                                                                                          | `output/var_lm.dta`                                          |
| `var_rpu.do`               | Runs VAR using aggregate regulatory uncertainty index.                                                                                                        | `output/var_rpu.dta`                                         |


### Table D2.3: Visualization Code (`code/visualization`)

| Script Name            | Description                                                                                  | Output                                                    |
|------------------------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `main_figures.py`      | Plots all figures in the main body of the paper.                                             | `figures/Figure[1-6].jpg`                                 |
| `appendix_figures.py`  | A master file to execute all Python scripts in `code/visualization/appendix_figure_scripts`. | See outputs of individual scripts below.                  |
| `appendix_figures_aggregate_index.py` | Plots appendix figures using aggregate indexes.                                              | `figures/appendix_figures/Appendix[E*].jpg`               |
| `appendix_figures_irf_aggregate.py` | Plots appendix figures using impulses responses from aggregate indexes.                      | `figures/appendix_figures/Appendix[F/G*/H*/I/J*/K/L].jpg` |
| `appendix_figures_categorical_index.py` | Plots appendix figures using categorical indexes.                                            | `figures/appendix_figures/Appendix[P/Q*].jpg`             |
| `appendix_figures_irf_categorical.py.py` | Plots appendix figures using impulses responses from categorical indexes.                    | `figures/appendix_figures/Appendix[R*].jpg`               |

---

## Computational Requirements

### Software Requirements

- The replication package contains one or more programs to install all dependencies and set up the necessary directory structure.  
- **Python 3.11–3.13**
  - A virtual Python environment should be created using `requirements.txt` in the home directory.  
  - Please run:  
    ```bash
    pip install -r requirements.txt
    ```  
  - See the [pip user guide on ensuring repeatability](https://pip.pypa.io/en/stable/user_guide/#ensuring-repeatability) for further instructions on creating and using the `requirements.txt` file.  
- **Stata (last run with version 14)**

---

## Instructions to Replicators

1. Download the replication package. 
2. Set up environment (see above).
3. Run `code/text_analysis/text_analysis_master.py` to reproduce indexes (optional).
4. Run `code/econometric_analysis/prepare_economic_data.py` to prepare data for econometric analysis.
5. Run `code/econometric_analysis/econometric_master.do` to implement econometric analysis.
6. Run `code/visualization/main_figures.py` to produce main figures.
7. Run `code/visualization/appendix_figures.py` to produce appendix figures.

---

## References

- Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic policy uncertainty. *Quarterly Journal of Economics*, 131(4), 1593–1636.
- Board of Governors of the Federal Reserve System (2022). *Federal Funds Effective Rate*. Retrieved February 10, 2022, from https://fred.stlouisfed.org/series/FEDFUNDS
- Board of Governors of the Federal Reserve System (2022). *Industrial Production*. Retrieved February 10, 2022, from https://fred.stlouisfed.org/series/INDPRO
- Cboe Exchange, Inc. (2022). *CBOE Volatility Index (VIX)*. Retrieved February 10, 2022, from https://www.cboe.com/tradable_products/vix/vix_historical_data/
- Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35–65.
- ProQuest (2022). Database accessed through ProQuest TDM Studio. Retrieved February 10, 2022, from [https://tdmstudio.proquest.com/home](https://tdmstudio.proquest.com/home). 
- Shapiro, A. H., Sudhof, M., & Wilson, D. J. (2022). Measuring news sentiment. *Journal of Econometrics*, 228(2), 221–243.
- S&P Dow Jones Indices LLC (2022). *S&P 500 Index, Monthly Close Price*. Retrieved February 10, 2022, from https://finance.yahoo.com/quote/%5EGSPC/history/?frequency=1mo
- Stone, P. J., Dunphy, D. C., Smith, M. S., & Ogilvie, D. M. (1966). *The General Inquirer: A Computer Approach to Content Analysis*. MIT Press.
- University of Michigan (2022). *Surveys of Consumers: Index of Consumer Sentiment*. Retrieved February 10, 2022, from https://data.sca.isr.umich.edu/
- U.S. Bureau of Economic Analysis (2022). *Real Gross Domestic Product*. Retrieved February 10, 2022, from https://fred.stlouisfed.org/series/GDPC1
- U.S. Bureau of Economic Analysis (2022). *Real Gross Private Domestic Investment*. Retrieved February 10, 2022, from https://fred.stlouisfed.org/series/GPDIC1
- U.S. Bureau of Labor Statistics (2022). *Consumer Price Index for All Urban Consumers: All Items in U.S. City Average*. Retrieved April 11, 2022, from https://fred.stlouisfed.org/series/CPIAUCSL
- U.S. Bureau of Labor Statistics (2022). *All Employees, Total Nonfarm*. Retrieved February 10, 2022, from https://fred.stlouisfed.org/series/PAYEMS
- Young, L., & Soroka, S. (2012). Affective news: The automated coding of sentiment in political texts. *Political Communication*, 29(2), 205–231.
