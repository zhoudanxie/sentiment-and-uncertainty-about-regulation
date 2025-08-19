import sys, subprocess

scripts = ["1_parse_xml.py",
           "2_extract_reg_sections.py",
           "3_identify_relevance.py",
           "4_sentiment_analysis.py",
           "5_categorize_articles.py",
           "6_construct_aggregate_index.py",
           "7_construct_categorical_index.py"]

for s in scripts:
    print(f"Running {s} ...")
    subprocess.run([sys.executable, s], check=True)  # check=True raises error if script fails