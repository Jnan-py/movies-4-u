import streamlit as st 
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 
import os 
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import io
from PIL import Image
import matplotlib.pyplot as plt
import urllib.parse

def main():

    st.set_page_config(layout = 'wide', page_title = "Movie Recommendation System")
    st.title('Movies 4 U')

    st.sidebar.title("Movies 4 U")
    st.sidebar.header("Pages")

    with st.sidebar.expander('PAGES', expanded = True):
        st.header('Go To')
        opt = st.selectbox(
            "Page",
            ['Search Movie',
            'Top Movies',
            'Movie Based',
            'User Based (study purpose)',] )

    d = r"datasets\movie_poster.csv"
    dd = pd.read_csv(d,names=['movie_id','url'])  
    def poster(movie_id):
        try:
            pst = dd[dd['movie_id']==movie_id]['url'].values[0]
            resp = requests.get(pst)
            img = Image.open(io.BytesIO(resp.content))
        except Exception as e:
            img = plt.imread(r"Pictures\Image-not-found.png")
        return img

    def fetch_poster(movie_id):        
        try:
            response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
            data = response.json()
            ff = "https://image.tmdb.org/t/p/w500/" + data['poster_path']

        except Exception as e:
            ff = plt.imread(r"Pictures\Image-not-found.png")

        return ff

    def get_link(query):
        base_url = "https://en.wikipedia.org/wiki/"
        encoded_query = urllib.parse.quote_plus(query.replace(" ", "_"))
        return f"{base_url}{encoded_query}"

    def idd_n(json_like_string):
        try:
            if len(json_like_string)!=2:
                json_string_fixed = json_like_string.replace("'", '"')
                dd = json.loads(json_string_fixed)
                return str([i['name'] for i in dd])
            else:
                return 'None'
        except Exception as e:
            return '[]'

    if opt == 'Search Movie':
        st.header('Search For Movie')
        data_mb1 = pd.read_csv(r"datasets\10000 Movies Data\10000 Movies Data")
        f_m = st.selectbox('Enter Movie Name: ',data_mb1['title'])
        if st.button('Search'):
            with st.spinner("Getting Results"):
                cols = st.columns([1,3])
                img = fetch_poster(data_mb1[data_mb1['title']==f_m]['Movie_id'].values[0])
                summ = data_mb1[data_mb1['title']==f_m]['overview'].values[0]
                gen = idd_n(data_mb1[data_mb1['title']==f_m]['Genres'].values[0])
                cols[0].image(img)
                cols[1].header(f_m)
                cols[1].subheader('**Summary**')
                cols[1].markdown(f'{summ}')
                cols[1].subheader('**Genre**')
                cols[1].markdown(gen)
                link = get_link(f_m)
                link_id = f"More Info about {f_m}"
                button_code = f"<a href='{link}' target='_blank' style='display: inline-block; padding: 10px 20px; background-color: #1D77BF; color: white; text-align: center; text-decoration: none; margin: 4px 2px; cursor: pointer; border-radius: 10px;'>{link_id}</a>"
                cols[1].markdown(button_code, unsafe_allow_html=True)
                st.markdown("""<hr style='border-top:1px solid blue;'>""",unsafe_allow_html=True)

                st.subheader('See Also')

                def idd(json_like_string):
                    try:
                        if len(json_like_string)!=2:
                            json_string_fixed = json_like_string.replace("'", '"')
                            dd = json.loads(json_string_fixed)
                            return str([i['id'] for i in dd])
                        else:
                            return 'None'
                    except Exception as e:
                        return '[]'

                data_mb1['gen_id'] = data_mb1['Genres'].apply(idd)
                data_mb1['key_id'] = data_mb1['Keywords'].apply(idd)
                data_mb1['f_ov'] = data_mb1['overview']+data_mb1['gen_id']+data_mb1['key_id']

                cv = CountVectorizer(max_features=1000, stop_words = 'english')
                vector = cv.fit_transform(data_mb1['f_ov'].values.astype('U')).toarray()

                sim_mat = cosine_similarity(vector)
                sim_df = pd.DataFrame(sim_mat,index=data_mb1['Movie_id'],columns=data_mb1['Movie_id'])

            def find_sim(title):
                id = data_mb1[data_mb1['title']==title]['Movie_id']
                movs=[]
                mov_posters = []
                ids = sim_df[id.values[0]].sort_values(ascending=False)[1:6]
                for i in ids.index:
                    m = data_mb1[data_mb1['Movie_id']==i]['title']
                    movs.append(m.values[0])
                    mov_posters.append(fetch_poster(i))
                return movs,mov_posters

            cc = st.columns(5)
            movs,post = find_sim(f_m)
            for i in range(5):
                cc[i].image(post[i])
                cc[i].write(f'**{movs[i]}**')

    elif opt=='Top Movies':
        st.header('Top Movies')
        st.subheader('Based on Rating')
        num = st.slider('Select number of Movies : ',min_value=5, max_value=500,step=1)
        
        if st.button('Go'):
            with st.spinner("Getting Movies"):
                data_mb1 = pd.read_csv(r"datasets\10000 Movies Data\10000 Movies Data")
                data_mb = data_mb1[['Movie_id','title','vote_average']]

                lst = data_mb.sort_values(by=['vote_average'],ascending=False)[:num]
                lst_ = lst['title'].values

                k_=0
                for row in range(num):
                    st.markdown("""<hr style='border-top:1px solid blue;'>""",unsafe_allow_html=True)
                    cols = st.columns([1,3])
                    if lst_[k_] in data_mb['title'].values:
                        img = fetch_poster(data_mb[data_mb['title']==lst_[k_]]['Movie_id'].values[0])
                        summary = data_mb1[data_mb1['title']==lst_[k_]]['overview'].values[0]
                        genres = data_mb1[data_mb1['title']==lst_[k_]]['Genres'].apply(idd_n)
                        key = data_mb1[data_mb1['title']==lst_[k_]]['Keywords'].apply(idd_n)

                    cols[0].image(img, use_container_width = True)
                    cols[1].subheader(f"**{lst_[k_]}**")
                    cols[1].write('')
                    cols[1].write('**Summary** : ')
                    cols[1].write(summary)
                    cols[1].write(f'**Genre** : {genres.values[0]}')
                    cols[1].write(f"**Keywords** : {key.values[0]}")

                    link = get_link(lst_[k_])
                    link_id = f"More Info about {lst_[k_]}"
            
                    button_code = f"<a href='{link}' target='_blank' style='display: inline-block; padding: 10px 20px; background-color: #1D77BF; color: white; text-align: center; text-decoration: none; margin: 4px 2px; cursor: pointer; border-radius: 10px;'>{link_id}</a>"
                    k_+=1
                    cols[1].markdown(button_code, unsafe_allow_html=True)
                    st.markdown("""</hr>""",unsafe_allow_html = True)
                    

    elif opt == 'Movie Based':
        st.header('Movie Based Recommender')        

        data_mb = pd.read_csv(r"datasets\10000 Movies Data\10000 Movies Data")

        wanted_cols = ['Movie_id','title','Genres','Keywords','overview']
        f_d_mb = data_mb[wanted_cols]

        mov = st.selectbox("**Enter the movie name :** ", f_d_mb['title'])

        def idd(json_like_string):
            try:
                if len(json_like_string)!=2:
                    json_string_fixed = json_like_string.replace("'", '"')
                    dd = json.loads(json_string_fixed)
                    return str([i['id'] for i in dd])
                else:
                    return 'None'
            except Exception as e:
                return '[]'

        f_d_mb['gen_id'] = f_d_mb['Genres'].apply(idd)
        f_d_mb['key_id'] = f_d_mb['Keywords'].apply(idd)
        f_d_mb['f_ov'] = f_d_mb['overview']+f_d_mb['gen_id']+f_d_mb['key_id']

        cv = CountVectorizer(max_features=1000, stop_words = 'english')
        vector = cv.fit_transform(f_d_mb['f_ov'].values.astype('U')).toarray()

        sim_mat = cosine_similarity(vector)
        sim_df = pd.DataFrame(sim_mat,index=f_d_mb['Movie_id'],columns=f_d_mb['Movie_id'])

        def find_sim(title):
            id = f_d_mb[f_d_mb['title']==title]['Movie_id']
            movs=[]
            mov_posters = []
            ids = sim_df[id.values[0]].sort_values(ascending=False)[1:37]
            for i in ids.index:
                m = f_d_mb[f_d_mb['Movie_id']==i]['title']
                movs.append(m.values[0])
                mov_posters.append(fetch_poster(i))
            return movs,mov_posters

        if st.button('Search'): 
            with st.spinner("Getting Movies"):
                ans,pos= find_sim(mov)
                rows = int(len(ans) / 4) + (1 if len(ans) % 4 != 0 else 0)
                k=0
                for i in range(rows):
                    cols = st.columns(4)
                    for j in range(4):
                        index = i * 4 + j
                        if index < len(ans):
                            cols[j].image(pos[k], use_container_width=True)
                            cols[j].write(f"**{ans[k]}**")
                            summary = data_mb[data_mb['title']==ans[k]]['overview'].values[0]
                            genres = data_mb[data_mb['title']==ans[k]]['Genres'].apply(idd_n)
                            with st.expander(f"Know About {ans[k]}", expanded = False):
                                cols[j].write("**Summary** : ")
                                if len(summary) > 250:
                                    cols[j].write(f"{summary[:250]}.....")
                                else:
                                    cols[j].write(summary)
                                cols[j].write(f"**Genre** : {genres.values[0]}")
                                link = get_link(ans[k])
                                link_id = f"More Info about {ans[k]}"                            
                                button_code = f"<a href='{link}' target='_blank' style='display: inline-block; padding: 10px 20px; background-color: #1D77BF; color: white; text-align: center; text-decoration: none; margin: 4px 2px; cursor: pointer; border-radius: 10px;'>{link_id}</a>"
                                cols[j].markdown(button_code, unsafe_allow_html=True)
                            k+=1
                        else:
                            cols[j].empty()

    elif opt == 'User Based (study purpose)':
        st.header('User Based Recommender')
        st.subheader('**Based on "ml-100K" dataset (Not Recommended)**') 
        
        u_d = r'datasets\ml-100k'

        cls = [
        'movie_id','movie_name','release_date','video_release_date','imdb_url','unknown','action','adventure','animation','children',
        'comedy','crime','documentary','drama','fantasy','film-noir','horror','musical','mystery','romance','sci-fi','thriller','war',
        'western'
    ]
        d_ms = pd.read_csv(os.path.join(u_d,'u.item'),sep='|',encoding='latin-1',names=cls)
        d_ms = d_ms[['movie_id','movie_name']]

        d_rts = pd.read_csv(os.path.join(u_d,'u.data'),names=['user_id','movie_id','rating','time_stamp'],sep='\t',encoding='latin-1')
        d_rts = d_rts.drop(columns=['time_stamp'])

        u_id = st.selectbox('Enter your user ID : ',d_rts['user_id'].unique())

        mat = d_rts.pivot(index = 'user_id' , columns = 'movie_id',values = 'rating')
        simm = mat.copy().fillna(0)

        cs = cosine_similarity(simm)
        csd = pd.DataFrame(cs,index=simm.index,columns=simm.index)

        def cf_knn(user_id,movie_id):
            if movie_id in mat:
                sim_scores = csd[user_id].copy()
                movie_ratings = mat[movie_id].copy()
                none_rating_index = movie_ratings[movie_ratings.isnull()].index
                movie_ratings = movie_ratings.drop(none_rating_index)
                sim_scores = sim_scores.drop(none_rating_index)

                mean_rating = np.dot(sim_scores,movie_ratings)/sim_scores.sum()
                
            return mean_rating

        def recom_movies_cf(user_id,n):
            user_movie = mat.loc[user_id].copy()
            for movie in mat:
                if pd.notnull(user_movie.loc[movie]):
                    user_movie.loc[movie] = 0
                else:
                    user_movie.loc[movie] = cf_knn(user_id,movie)
            movie_sort = user_movie.sort_values(ascending=False)[:n]
            recom_movies = d_ms.loc[movie_sort.index]
            recommendations = recom_movies['movie_name']
            recom_ids = recom_movies['movie_id']
            posters = [poster(ids) for ids in recom_ids]
            return recommendations,posters
        
        if st.button('Search'):
            with st.spinner("Getting Movies"):
                ans,pos = recom_movies_cf(int(u_id),36)
                rows = int(len(ans) / 4) + (1 if len(ans) % 4 != 0 else 0)
                k=0
                for i in range(rows):
                    cols = st.columns(4)
                    for j in range(4):
                        index = i * 4 + j
                        if index < len(ans):
                            cols[j].image(pos[k], use_container_width=True)
                            cols[j].write(ans.iloc[k])
                            k+=1
                        else:
                            cols[j].empty()

if __name__ == "__main__":
    main()

                
