import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


def preprocess_text(text):
   
    tokens = nltk.word_tokenize(text)
    
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [word for word in tokens if word.lower() not in stopwords and word.isalnum()]
    return " ".join(tokens)


def check_plagiarism(text1, text2):
    
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    

    return cosine_sim[0][0] * 100


if __name__ == "__main__":
   
    text1 = """Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans."""
    text2 = """AI refers to the ability of a machine to perform tasks that would typically require human intelligence, such as learning, reasoning, and problem-solving."""
    
    
    similarity = check_plagiarism(text1, text2)
    
    print(f"The similarity between the texts is: {similarity:.2f}%")
    
    if similarity > 80:
        print("Warning: High similarity detected! Possible plagiarism.")
    else:
        print("The texts are sufficiently different.")
