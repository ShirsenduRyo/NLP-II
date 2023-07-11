import newspaper
import requests

class ArticleExtractor:
    def __init__(self, url):
        self.url = url
        self.article = None

    def download_article(self):
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            self.article = newspaper.Article(self.url)
            self.article.download()
            self.article.parse()
        except (requests.exceptions.RequestException, newspaper.article.ArticleException) as e:
            print(f"Error occurred while downloading the article: {e}")

    def extract_title(self):
        return self.article.title if self.article else None

    def extract_text(self):
        return self.article.text if self.article else None

    def extract_authors(self):
        return self.article.authors if self.article else None

    def extract_publish_date(self):
        return self.article.publish_date if self.article else None