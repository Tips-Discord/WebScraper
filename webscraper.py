import aiohttp
import asyncio
import os
import random
from bs4 import BeautifulSoup
from colorama import init, Fore
from googlesearch import search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import validators

init(autoreset=True)
debug = False

class WebScraperQA:
    def __init__(self, max_websites=50, num_results=25):
        self.documents = []
        self.sources = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = None
        self.scraped_urls = set()
        self.max_websites = max_websites
        self.num_results = num_results
        self.visited_count = 0
        self.tried_urls = 0

    @staticmethod
    def clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')

    @staticmethod
    def generate_random_headers():
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        accept_languages = [
            'en-US,en;q=0.9',
            'en-GB,en;q=0.9',
            'en-CA,en;q=0.9',
            'en-AU,en;q=0.9',
            'en-NZ,en;q=0.9'
        ]
        return {
            'User-Agent': random.choice(user_agents),
            'Accept-Language': random.choice(accept_languages),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.google.com',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1'
        }

    async def fetch(self, url, session, headers):
        try:
            async with session.get(url, headers=headers, timeout=20) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientError as e:
            if debug:
                print(Fore.RED + f"HTTP error occurred while fetching {url}: {e}")
            return None

    async def scrape_website(self, url, session):
        if url in self.scraped_urls or not validators.url(url):
            if debug:
                print(Fore.RED + f"Invalid or duplicate URL: {url}")
            return None, None

        self.tried_urls += 1

        if not await self.is_allowed_by_robots_txt(url, session):
            if debug:
                print(Fore.RED + f"Disallowed by robots.txt: {url}")
            return None, None

        headers = self.generate_random_headers()
        html = await self.fetch(url, session, headers)
        if html:
            self.scraped_urls.add(url)
            if debug:
                print(Fore.GREEN + f"Scraped from: {url}")
            return BeautifulSoup(html, 'html.parser'), url

        return None, None

    async def is_allowed_by_robots_txt(self, url, session):
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        headers = self.generate_random_headers()
        try:
            async with session.get(robots_url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    return self.parse_robots_txt(content, parsed_url.path)
        except Exception:
            return True

    @staticmethod
    def parse_robots_txt(content, path):
        rules = [line.strip() for line in content.splitlines() if line.strip()]
        user_agent, allowed = '*', True
        for rule in rules:
            if rule.lower().startswith('user-agent:'):
                user_agent = rule.split(':', 1)[1].strip()
            if user_agent == '*' and rule.lower().startswith('disallow:'):
                disallowed_path = rule.split(':', 1)[1].strip()
                if path.startswith(disallowed_path):
                    allowed = False
        return allowed

    @staticmethod
    def extract_text(soup):
        for element in soup(["script", "style", "header", "footer", "meta", "noscript", "img"]):
            element.decompose()
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs if p.get_text(strip=True))

        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description and 'content' in meta_description.attrs:
            text = f"{meta_description['content']} {text}"

        return text

    @staticmethod
    def filter_text(text):
        sponsor_keywords = [
            'sponsor', 'ad', 'advertisement', 'partner', 
            'promoted', 'newsletter', 'subscribe', 'follow us', 
            'cookie policy'
        ]
        sentences = text.split('. ')
        filtered_sentences = [
            sentence for sentence in sentences 
            if len(sentence) > 50 and not any(keyword in sentence.lower() for keyword in sponsor_keywords)
        ]
        return '. '.join(filtered_sentences)

    def add_document(self, text, source):
        filtered_text = self.filter_text(text)
        if filtered_text:
            self.documents.append(filtered_text)
            self.sources.append(source)
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    async def process_url(self, url, session):
        soup, source = await self.scrape_website(url, session)
        if soup:
            text = self.extract_text(soup)
            self.add_document(text, source)
            self.visited_count += 1
            if not debug:
                print(f"\rScraped: {self.visited_count}", end='')

    async def auto_scrape_and_learn(self, urls):
        async with aiohttp.ClientSession() as session:
            tasks = [self.process_url(url, session) for url in urls[:self.max_websites]]
            await asyncio.gather(*tasks)

    def search_and_learn(self, query):
        try:
            search_results = list(search(query, stop=self.num_results))
            asyncio.run(self.auto_scrape_and_learn(search_results))
        except Exception as e:
            if debug:
                print(Fore.RED + f"Error during search and learn: {e}")

    def answer_question(self, question):
        if not self.documents:
            if debug:
                print(Fore.RED + "No websites read")
            return "No relevant answer found.", []

        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.doc_vectors).flatten()
        most_similar_idx = similarities.argmax()

        if similarities[most_similar_idx] > 0.3:
            return self.documents[most_similar_idx], [self.sources[i] for i in similarities.argsort()[-5:][::-1]]

        return "No relevant answer found.", []

if __name__ == "__main__":
    qa_system = WebScraperQA()
    qa_system.clear_screen()
    question = input("Question: ")
    qa_system.clear_screen()
    print(Fore.CYAN + f"Question: {question}\n")
    print(Fore.GREEN + "Searching and learning...")
    qa_system.search_and_learn(question)
    print(Fore.GREEN + "\nAnswering the question...")
    answer, sources = qa_system.answer_question(question)
    if answer:
        print(Fore.BLUE + f"\nAnswer: {answer}")
        if sources:
            print(Fore.CYAN + "\nSources:")
            for source in sources:
                print(Fore.CYAN + source)
    else:
        print(Fore.RED + "No answer could be found.")
    print(Fore.CYAN + f"\nTried URLs: {qa_system.tried_urls}")
    print(Fore.CYAN + f"\nProcessed URLs: {qa_system.visited_count}")
