# WebScraper

## Overview

WebScraper is an advanced web scraping and question-answering system written in Python. It combines the power of asynchronous web scraping, natural language processing, and machine learning to search the web for relevant information and provide concise answers to user queries.

## Features

- **Asynchronous Web Scraping**: Uses `aiohttp` and `asyncio` for efficient and fast web scraping.
- **Natural Language Processing**: Utilizes `BeautifulSoup` for parsing HTML and `sumy` for text summarization.
- **Machine Learning**: Implements TF-IDF vectorization and cosine similarity from `scikit-learn` for document comparison and relevance scoring.
- **Dynamic User-Agent**: Leverages `fake_useragent` to generate random user agents for scraping.
- **Search Integration**: Integrates with Google Search to find relevant websites based on the user's query.
- **Content Filtering**: Filters out unwanted content like advertisements and promotional material.

## Installation

### Prerequisites

- Python 3.11.4 or higher
- Pip package manager

### Dependencies

Install the required Python packages using pip:

```sh
pip install -r requirements.txt
```

### Download the Project

Clone the repository or download the script directly:

```sh
git clone https://github.com/Tips-Discord/WebScraper
cd WebScraper
```

## Usage

### Running the Program

To run the WebScraperQA program, execute the following command in your terminal:

```sh
python webscraper.py
```

### Interacting with the Program

Upon execution, the program will prompt you to enter a question:


### Example

```sh
Question: What is the capital of France?

Searching and learning...

Websites Visited: 10

Answering the question...

Answer: The capital of France is Paris.
```


## License

This project is licensed under the GNU General Public License v2.0.

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
