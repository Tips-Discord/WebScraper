import aiohttp
import asyncio
import os
import random
import sys
import validators

from bs4 import BeautifulSoup
from colorama import init, Fore, Style
from fake_useragent import UserAgent
from googlesearch import search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

init(autoreset=True)

class WebScraperQA:
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = None
        self.scraped_urls = set()
        self.ua = UserAgent()
        self.visited_count = 0
        self.max_websites = 15

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    async def scrape_website(self, url, session):
        if url in self.scraped_urls or not validators.url(url):
            return None
        try:
            headers = {'User-Agent': self.ua.random}
            async with session.get(url, headers=headers, timeout=10) as response:
                response.raise_for_status()
                self.scraped_urls.add(url)
                soup = BeautifulSoup(await response.text(), 'html.parser')
                return soup
        except Exception as e:
            return None

    def extract_text(self, soup):
        for script in soup(["script", "style", "header", "footer", "meta", "noscript", "img"]):
            script.decompose()
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs if p.get_text(strip=True)])

        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description and 'content' in meta_description.attrs:
            text = meta_description['content'] + " " + text

        text = text + " " + soup.prettify()[:3500]

        return text

    def filter_text(self, text):
        sponsor_keywords = ['sponsor', 'ad', 'advertisement', 'partner', 'promoted', 'newsletter', 'subscribe', 'follow us', 'cookie policy']
        sentences = text.split('. ')
        filtered_sentences = [sentence for sentence in sentences if len(sentence) > 50 and not any(keyword in sentence.lower() for keyword in sponsor_keywords)]
        return '. '.join(filtered_sentences)

    def summarize_text(self, text):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, 5)
        return ' '.join([str(sentence) for sentence in summary])

    def add_document(self, text, url=None):
        filtered_text = self.filter_text(text)
        summarized_text = self.summarize_text(filtered_text)
        if summarized_text:
            self.documents.append(summarized_text)
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    async def process_url(self, url, session):
        await asyncio.sleep(random.uniform(1, 2))
        soup = await self.scrape_website(url, session)
        if soup:
            text = self.extract_text(soup)
            self.add_document(text, url=url)
            self.visited_count += 1
            sys.stdout.write(f"\rWebsties Visited: {self.visited_count}")
            sys.stdout.flush()

    async def auto_scrape_and_learn(self, urls):
        async with aiohttp.ClientSession() as session:
            tasks = [self.process_url(url, session) for url in urls[:self.max_websites]]
            await asyncio.gather(*tasks)

    def search_and_learn(self, query, num_results=15):
        try:
            search_results = list(search(query, num_results=num_results))
            asyncio.run(self.auto_scrape_and_learn(search_results))
        except Exception as e:
            pass

    def answer_question(self, question):
        if not self.documents:
            pass
            return "No relevant answer found."
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.doc_vectors).flatten()
        most_similar_idx = similarities.argmax()
        if similarities[most_similar_idx] > 0.09:
            return self.documents[most_similar_idx]
        else:
            return "No relevant answer found."

if __name__ == "__main__":
    qa_system = WebScraperQA()
    question = input("Question: ")
    qa_system.clear_screen()
    print(Fore.CYAN + f"Fixed Question: {question}\n")
    print(Fore.GREEN + "Searching and learning...")
    qa_system.search_and_learn(question)
    print(Fore.GREEN + "\nAnswering the question...")
    answer = qa_system.answer_question(question)
    if answer:
        print(Fore.BLUE + f"\nAnswer: {answer}")
    else:
        print(Fore.RED + "No answer could be found.")
