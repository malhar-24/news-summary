from flask import Flask, render_template, request, jsonify
import nltk
from langchain_groq import ChatGroq
from newspaper import Article
from newspaper.article import ArticleException

# Ensure 'punkt' tokenizer is available
nltk.download('punkt')

app = Flask(__name__)

# Function to summarize the news
def summarize_news(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        return {
            'title': article.title,
            'authors': article.authors,
            'publish_date': article.publish_date,
            'summary': article.summary
        }
    except ArticleException as e:
        return {'error': f"Failed to download or process the article: {e}"}
    except Exception as e:
        return {'error': f"An error occurred: {e}"}

# Function to ask a question about the summary
def ask_question_about_summary(summary, question):
    prompt = f"Based on this summary: {summary}\n\n{question}"
    llm = ChatGroq(
        temperature=0,
        groq_api_key='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',
        model_name="llama-3.1-70b-versatile"
    )
    response = llm.invoke(prompt)
    return response.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    url = request.form.get('url')
    summary_data = summarize_news(url)
    if 'error' in summary_data:
        return jsonify({'error': summary_data['error']})
    return jsonify(summary_data)

@app.route('/ask_question', methods=['POST'])
def ask_question():
    summary = request.form.get('summary')
    question = request.form.get('question')
    response = ask_question_about_summary(summary, question)
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)

