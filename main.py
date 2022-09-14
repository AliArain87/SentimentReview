import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px
from PIL import Image
from annotated_text import annotated_text
from streamlit_option_menu import option_menu
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pickle



df = pd.read_csv('a2_RestaurantReviews_FreshDump.tsv', sep='\t', quoting=3)


#Functions 

def data_proc():
        nltk.download('stopwords')

        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        ps = PorterStemmer()

        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')

        corpus=[]

        for i in range(0, 100):
            review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
            review = review.lower()
            review = review.split()
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            corpus.append(review)
        
        
        st.markdown('---')
        st.title('Data transformation')
        
        # Loading BoW dictionary
        
        cvFile='c1_BoW_Sentiment_Model.pkl'
        cv = pickle.load(open(cvFile, "rb"))
        X_fresh = cv.transform(corpus).toarray()
        
        

def pred_fun():
    nltk.download('stopwords')

    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus=[]

    for i in range(0, 100):
        review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
        
    cvFile='c1_BoW_Sentiment_Model.pkl'
    cv = pickle.load(open(cvFile, "rb"))
    X_fresh = cv.transform(corpus).toarray()
    
    import joblib
    classifier = joblib.load('c2_Classifier_Sentiment_Model')

    y_pred = classifier.predict(X_fresh)
    st.markdown('---')
    st.title('✨Results✨')
    st.write(y_pred)
    
    
    df['predicted_label'] = y_pred.tolist()
    st.markdown('---')
    st.title('✨Dataframe with predicted label✨')

    # color the dataframe
    def color_df(val):
        if val==0:
            color = 'red'
        else:
            color = 'blue'
        return f'background-color: {color}'

    df1 = pd.DataFrame(df, columns=['Review', 'predicted_label'])
    st.dataframe(df1.style.applymap(color_df, subset=['predicted_label']))

        





def vis():
    nltk.download('stopwords')

    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus=[]

    for i in range(0, 100):
        review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
        
    cvFile='c1_BoW_Sentiment_Model.pkl'
    # cv = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open('./drive/MyDrive/Colab Notebooks/2 Sentiment Analysis (Basic)/3.1 BoW_Sentiment Model.pkl', "rb")))
    cv = pickle.load(open(cvFile, "rb"))
    X_fresh = cv.transform(corpus).toarray()
    
    import joblib
    classifier = joblib.load('c2_Classifier_Sentiment_Model')
    y_pred = classifier.predict(X_fresh)
    df['predicted_label'] = y_pred.tolist()
    df1 = pd.DataFrame(df, columns=['Review', 'predicted_label'])

    group1 = df1.groupby(['predicted_label']).count()
    fig = px.histogram(df, x="predicted_label", color="predicted_label",title="Predicted Sentiments",labels={"predicted_label": "Sentiment"}, 
                   nbins=2, opacity=0.8).update_layout( xaxis = dict(tickmode = 'array', tickvals = [0, 1], ticktext = ['Negative', 'Positive']), )

    st.markdown('---')
    st.title('Histogram of predicted sentiments')
    st.write(fig)

    pie_chart=px.pie(group1, values='Review', names='Review', hover_name='Review', title='Predicted Sentiments', 
        color_discrete_sequence= ['#f00707','yellow'], color='Review', 
        hole=0.3, labels={'Review':'Sentiment'}).update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, 
                                                                            xanchor="right", x=1), title_x=0.5, title_y=0.9, title_font_size=30)
        
      
    st.markdown('---')
    st.title('Pie chart of predicted sentiments')
    st.write(pie_chart)
    
    st.markdown('---')
    ali6 = Image.open('ali6.png')
    st.image(ali6, width=300)
    st.title('🟡Suggestions Form My App🟡')
    if group1['Review'][0] > group1['Review'][1]:
        st.markdown('---')
        st.markdown('### **⚠The overall reviews are negative⚠**')
        st.markdown('---')
    else:
        st.markdown('### **✅The overall reviews are positive✅**')






# Modify app name and Icon
# Config Function 
st.set_page_config(page_title='Ali Hasnain App', page_icon=':shark:', layout='wide', initial_sidebar_state='auto')


# Hide Menu and Footer
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """   
st.markdown(hide_menu_style, unsafe_allow_html=True)




st.title('✨Reviews App✨')
st.markdown('### ✨Sentiment Analysis✨')
st.markdown('---')







selected = option_menu(
    menu_title='My App (Ali Hasnain)✨',
    options= ['ABOUT APP', 'DATASET', 'PREDICITION', 'VISUALIZATION'],
    icons = ['book-half', 'file-earmark-code-fill', 'graph-up-arrow', 'pie-chart-fill'],
    
    menu_icon = 'toggle-on',
    default_index = 0,
    orientation = 'horizontal',
)

    

if selected == 'ABOUT APP':
    st.title(f'✨About App✨')
    st.markdown('---')
    img1 = Image.open('ali.png')
    img2 = Image.open('ali2.png')
    col1, col2 = st.columns(2)
    with col2:
        st.image([img2,img1], width=200)
    with col1:
        st.header('The app is created by Ali Hasnain')
        st.write('''This app will let you help to predict the sentiment of the review.
                 As there are many organizations that are taking reviews from the customers
                 because they want to know the feedback from their customers and users.
                 So, this app will help them to predict the sentiment of the review.
                 As there are many reviews that are positive and some are negative.
                 This App will help them to watch the overall reviews with the tag of
                 positive (1) and negetive (0) reviews with interacitve graphs.''')
        st.markdown('---')
        st.header('💬CONTACT ME')
        st.markdown('##### 👉🏼[LinkedIn](https://www.linkedin.com/in/ali-hasnain-8b047a210/)  👉🏼[GitHub](https://github.com/AliArain87)')
        
        
    
    
       
    
elif selected == 'DATASET':
    st.title(f'✨Dataset✨')
    c1 , c2 = st.columns(2)
    with c1:
        st.dataframe(df)
    with c2:
        img3 = Image.open('ali3.png')
        st.image(img3, width=300)
        
    st.markdown('---')    
    st.title('✨Data Description✨')
    st.write("The rows of dataset =",df.shape[0])
    st.write("The column of dataset =",df.shape[1])
    
    st.markdown('---')
    ## **Data Preprocessing**
    st.title('✨Data Preprocessing✨')
    ali4 = Image.open('ali4.png')
    st.image(ali4)
    ali5 = Image.open('ali5.jpg')
    data_proc()
    st.image(ali5, width=400)
    
    
    
    
    
    
elif selected == 'PREDICITION':
    st.title(f'✨Prediction✨')
    data_proc()
    st.markdown('## 💁‍♂️**Predictions (via sentiment classifier)**')
    st.markdown('### 💁‍♂️**Using saved model (Trained) for new prediction**')
    pred_fun()
    
   
    
else:
    st.title(f'✨Visualization✨')
    vis()
    
    
    






# df.to_csv("D:\Big Data\Final Project\c3_Predicted_Sentiments_Fresh_Dump.tsv", sep='\t', encoding='UTF-8', index=False)


