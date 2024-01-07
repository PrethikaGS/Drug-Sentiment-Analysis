import streamlit as st
import pandas as pd

# Define functions for different pages
def predictions_page():
    st.title("Predictions")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import re
    from sklearn.svm import SVC
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

    data2 = pd.read_csv('D:\PSG\sem5\ml\ca2-hackathon\pre_processed_SA.csv')
    data2.head()

    data2['preprocessed_text'] = data2['preprocessed_text'].astype('string')
    X = data2["preprocessed_text"]  # Features
    y = data2['sentiment']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Vectorize the text data using CountVectorizer (binary=True)
    vectorizer = CountVectorizer(binary=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)


    # Vectorize the text data using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)


    from sklearn.naive_bayes import BernoulliNB, MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    # Train a logistic regression classifier
    logistic_classifier = LogisticRegression()
    logistic_classifier.fit(X_train_tfidf, y_train)



    def prediction(user_input):
        user_input_tokens = user_input.lower().split()
        # Load the TF-IDF vectorizer and transform the user input
        tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = tfidf_vectorizer.fit_transform(data2['preprocessed_text'])
        user_input_tfidf = tfidf_vectorizer.transform([' '.join(user_input_tokens)])
        # Load the logistic regression model (use the trained model)
        model = LogisticRegression()
        model.fit(X_tfidf, data2['sentiment'])
        print(user_input_tfidf)
        # Make a prediction on the user input
        user_input_prediction = model.predict(user_input_tfidf)
        print(user_input_prediction)
        # Interpret the prediction
        if user_input_prediction[0] == 0:
            sentiment = "negative"
        else:
            sentiment = "positive"
        return sentiment

    # Ask the user for their name
    userId = st.text_input("Enter your unique id :")
    userReview = st.text_input("Enter your review :")

    if st.button("Submit"):
        # Code to execute when the button is clicked
        ans = prediction(userReview)
        st.write("You submitted:", ans)


    
def reviews_page():
    st.title("Reviews")
    # Add content for the Reviews page here

def drugs_page():
    st.title("List of Drugs")
    # Load the dataset
    df = pd.read_csv('D:\PSG\sem5\ml\ca2-hackathon\sentimentAnalysis.csv')

    # Create a text input for entering a disease
    selected_disease = st.text_input("Enter a disease:")
    if(selected_disease!=""):

        # Filter the dataset based on the entered disease
        filtered_df = df[df['condition'].str.lower().str.contains(selected_disease.lower())]

        # Display the list of drugs for the selected disease
        st.title(f"List of Drugs for '{selected_disease}'")
        drugs_for_disease = filtered_df['drugName'].unique()
        st.write(drugs_for_disease)

    # Add content for the List of Drugs page here

def medicines_page():
    st.title("List of Diseases")
    # Load the dataset
    df = pd.read_csv('D:\PSG\sem5\ml\ca2-hackathon\sentimentAnalysis.csv')

    # Create a text input for entering a disease
    selected_disease = st.text_input("Enter a drug:")
    if(selected_disease!=""):

        # Filter the dataset based on the entered disease
        filtered_df = df[df['drugName'].str.lower().str.contains(selected_disease.lower())]

        # Display the list of drugs for the selected disease
        st.title(f"Conditions for which '{selected_disease}' is prescribed ")
        drugs_for_disease = filtered_df['condition'].unique()
        st.write(drugs_for_disease)

    # Add content for the List of Drugs page here

# Streamlit app
def main():
    st.title("Medical App")

    # Create a navigation menu
    page = st.selectbox("Select a page:", ("Predictions", "Reviews", "List of Drugs", "List of Diseases"))

    # Render the selected page
    if page == "Predictions":
        predictions_page()
    elif page == "Reviews":
        reviews_page()
    elif page == "List of Drugs":
        drugs_page()
    elif page == "List of Diseases":
        medicines_page()

if __name__ == "__main__":
    main()
