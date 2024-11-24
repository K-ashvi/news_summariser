from transformers import pipeline
from newspaper import Article

# Step 1: Load the Fine-Tuned Model
summarizer = pipeline("summarization", model="./fine_tuned_bart", tokenizer="./fine_tuned_bart")

# Step 2: Function to Fetch Article Content from a URL
def fetch_article_content(url):
    """
    Fetch the content of an article from a URL.
    Args:
        url (str): URL of the news article.
    Returns:
        str: Article text or None if fetching fails.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error fetching article: {e}")
        return None

# Step 3: Summarize the Article
def summarize_article(url):
    """
    Fetch and summarize a news article from the given URL.
    Args:
        url (str): URL of the news article.
    Returns:
        str: Summary of the article or error message.
    """
    article_content = fetch_article_content(url)
    if article_content:
        print("\nFetched Article Content:\n", article_content[:500], "...")  # Display the first 500 characters
        print("\nSummarizing...\n")
        try:
            summary = summarizer(article_content, max_length=130, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            return f"Error during summarization: {e}"
    else:
        return "Could not fetch the article. Please check the URL."

# Step 4: Input URL and Summarize
if __name__ == "__main__":
    url = input("Enter the URL of the news article: ")
    summary = summarize_article(url)
    print("\nSummary:\n", summary)
