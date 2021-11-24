#################################################################
# Projet 2 WCS (Novembre 2021)
# Equipe 4 : Aline, Elisa, Laurent, Nizar
# 
# Code Demo sur Streamlit
######################### DECLARATIONS ##########################
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import webbrowser
from sklearn.neighbors import NearestNeighbors

# Chemins d'accès
images_path = "D:\\Images\\"
CSV_path = "D:\\CSV\\"

##################### CHARGEMENT DES BASES #######################
# Path des 3 bases
base_movies_path = CSV_path + "base_movies.csv"
base_genres_path = CSV_path + "base_genres.csv"
base_names_path = CSV_path + "base_names.csv"

# Chargement base_movies
base_movies = pd.read_csv(base_movies_path, sep = ",")

# Chargement base_genres
base_genres = pd.read_csv(base_genres_path, sep = ",")

# Chargement base_names
base_names = pd.read_csv(base_names_path, sep = ",")

########################### FONCTIONS ###############################
# Fonction de recherche de films à partir d'un libellé partiel
# La fonction recherche toutes les lignes dans Movies, dont primaryTitle ou TitleFR contient le libéllé passé en paramètre
# Entrée : Dataframe Movies, 
#          Libéllé à rechercher
# Sortie : liste contenant la description succint de chaque film sélectionné
def recherche_film (df_movies, pattern):
    movies_list = []
    pattern = "(?i)" + pattern  # Expression regex qui indique qu'il faut ignorer la casse  
    recherche = df_movies[(df_movies['primaryTitle'].str.contains(pattern)) | (df_movies['originalTitle'].str.contains(pattern)) | (df_movies['titleFR'].str.contains(pattern))]
    if len(recherche) > 0:
      for r in range(len(recherche)):
        descriptif = recherche.iloc[r,0]
        descriptif += '  ' + recherche.iloc[r,2]
        descriptif += ' ('+ str(int(recherche.iloc[r,6])) + ')'
        movies_list.append(descriptif)
    return movies_list

# Fonction de récupération des informations complètes d'un film à partir de son code IMDb
# La fonction va rechercher les caractéristiques d'un film dans Movies, Genres, Names
# Entrée : Dataframe Movies, 
#          Dataframe Genres, 
#          Dataframe Names,
#          code IMDb
# Sortie : Dictionnaire contenant les caratéristiques du film
def fiche_film (df_movies, df_genres, tconst):
    dico_desc = {}
    if tconst in df_movies.tconst.values:
      indM =  df_movies.loc[df_movies['tconst'] ==  tconst].index
      genres = ''
      genre1 = df_movies.loc[indM,'genre1'].values[0]
      if not pd.isnull(genre1):
        indG1 =  df_genres.loc[df_genres['genre'] ==  genre1].index
        genres = df_genres.loc[indG1, ('genreFR')].values[0]
      genre2 = df_movies.loc[indM,'genre2'].values[0]
      if not pd.isnull(genre2):
       indG2 =  df_genres.loc[df_genres['genre'] ==  genre2].index
       genres += ', ' + df_genres.loc[indG2, ('genreFR')].values[0]
      genre3 = df_movies.loc[indM,'genre3'].values[0]
      if not pd.isnull(genre3):
        indG3 =  df_genres.loc[df_genres['genre'] ==  genre3].index
        genres += ', ' + df_genres.loc[indG3, ('genreFR')].values[0]
      dico_desc.update({'Code IMDb: ': df_movies.loc[indM,'tconst'].values[0]})
      if not pd.isnull(df_movies.loc[indM,'titleFR'].values[0]):
        dico_desc.update({'Titre: ': df_movies.loc[indM,'titleFR'].values[0]})
      else:
        dico_desc.update({'Titre: ': df_movies.loc[indM,'primaryTitle'].values[0]})
      dico_desc.update({'Année: ': str(int(df_movies.loc[indM,'startYear'].values[0]))})
      dico_desc.update({'Durée: ': str(int(df_movies.loc[indM,'runtimeMinutes'].values[0])//60) + 'h'+str(int(df_movies.loc[indM,'runtimeMinutes'].values[0])%60).zfill(2)})
      dico_desc.update({'Genre(s): ': genres})
      dico_desc.update({'Note: ': str(df_movies.loc[indM,'averageRating'].values[0]) + ' (' + str(int(df_movies.loc[indM,'numVotes'].values[0]))+' votes)'})
      # dico_desc.update({'Réalisateur(s): ': df_movies.iloc[indM,7].values[0]})   # ajouter un df names pour récupérer les noms des réalisateurs 
      # dico_desc.update({'Scénariste(s): ': df_movies.iloc[indM,8].values[0]})    # ajouter un df names pour récupérer les noms des scénaristes
      # ajouter le noms des acteurs principaux
      dico_desc.update({'Lien IMDb: ': 'https://www.imdb.com/title/' + df_movies.loc[indM,'tconst'].values[0]})      
    return dico_desc

# Fonction de récupération des informations succintes d'un film à partir de son code IMDb
# La fonction va rechercher les informations dans Movies
# Entrée : Dataframe Movies, 
#          code IMDb
# Sortie : String contenant le code IMDb et le titre du film
def descriptif_film (df_movies, tconst):
    descriptif = ''
    if tconst in df_movies.tconst.values:
      indM =  df_movies.loc[df_movies['tconst'] ==  tconst].index
      descriptif = df_movies.loc[indM,'tconst'].values[0]
      if not pd.isnull(df_movies.loc[indM,'titleFR'].values[0]):
        descriptif += '  ' + df_movies.loc[indM,'titleFR'].values[0]
      else:
        descriptif += '  ' + df_movies.loc[indM,'primaryTitle'].values[0]
      descriptif += ' ('+ str(int(df_movies.loc[indM,'startYear'].values[0])) + ')'
    return descriptif

# Fonction de 'formatage' du dataset pour le ML
# On met en place une série de fonctions qui permettent de :
#    1. Réduire le dataset des films à comparer
#          - on prend dans la base les films avec les mêmes genres que le film de référence
#          - on prend dans la base les films qui ont des réalisateurs en commun avec le film de référence
#          - on prend dans la base les films qui ont des acteurs en commun avec le film de référence
#          - on ne prend pas ceux avec des scénaristes pour réduire le temps de réponse
#          - on regroupe ces films pour n'en faire qu'une seule base (et on supprime les doublons)
#          - on réduit encore cette base aux films +/- 5 ans par rapport à la date du film de référence
        
#    2. Créer de nouvelles features pour la comparaison grâce aux données présentes dans le datasets
#       à partir de cette base réduite,on crée deux nouvelles colonnes qui nous permettront de comparer les films sur les réalisateurs et les acteurs
#         - 'similarity_d' : la nouvelle colonne renvoient 1 si le réalisateur du film de référence est dans la liste des réalisateurs, 0 s'il n'y est pas
#         - 'similarity_a' : la nouvelle colonne renvoient 1 si l'acteur principal du film de référence est dans la liste des réalisateurs, 0 s'il n'y est pas
#
# Ces fonctions sont compilées dans une seule fonction globale qui renvoit le dataset à utiliser pour le machine learning en fonction du film de référence entré par l'utilisateur 

def full_base_reduite(base_movies, tconst):
    ########  Sélection des films à comparer
    def select_genres(base_movies, tconst) :
        base_reduite_g = base_movies.drop(base_movies.index)
        liste = base_movies['genres'].str.split(',')[base_movies['tconst'] == tconst].values
        for genre in liste[0] :
            base_reduite_g = pd.concat([base_reduite_g, base_movies[base_movies['genres'].str.contains(genre)]])
        return base_reduite_g

    def select_director(base_movies, tconst) :
        base_reduite_d = base_movies.drop(base_movies.index)
        liste = base_movies['directors'].str.split(',')[base_movies['tconst'] == tconst].values
        for directors in liste[0] :
            base_reduite_d = pd.concat([base_reduite_d, base_movies[base_movies['directors'].str.contains(directors)]])
        return base_reduite_d
      
    def select_actor(base_movies, tconst) :
        base_reduite_a = base_movies.drop(base_movies.index)
        liste = base_movies['actors'].str.split(',')[base_movies['tconst'] == tconst].values
        for actors in liste[0] :
            base_reduite_a = pd.concat([base_reduite_a, base_movies[base_movies['actors'].str.contains(actors)]])
        return base_reduite_a    

    def select_year(base_movies, tconst) :
        year = base_movies['startYear'][base_movies['tconst'] == tconst].values
        base_reduite_y = base_movies[(base_movies['startYear'] >= year[0]-5) & (base_movies['startYear'] <= year[0]+5)]
        return base_reduite_y

    ########  Création d'un dataset réduit     
    base_reduite = pd.concat([select_genres(base_movies, tconst), select_director(base_movies, tconst), select_actor(base_movies, tconst)])
    base_reduite.drop_duplicates(inplace=True)
    base_reduite = select_year(base_reduite, tconst)

    ########  Ajout de nouvelles features
    #def similarity_actors(value):
    #    actor = base_movies['actor1'][base_movies['tconst'] == tconst].values
    #    if actor == value :
    #        return 1
    #    else :
    #        return 0

    #base_reduite['similarity_a'] = base_reduite['actors'].apply(similarity_actors)

    def similarity_directors(value):
        director = base_movies['director1'][base_movies['tconst'] == tconst].values
        if director in value : 
            return 1
        else :
            return 0

    base_reduite['similarity_d'] = base_reduite['directors'].str.split(',').apply(similarity_directors)

    #def similarity_writers(value):
    #   writer = base_movies['writer1'][base_movies['tconst'] == tconst].values
    #    if writer in value : 
    #        return 1
    #    else :
    #        return 0
    #base_reduite['similarity_w'] = base_reduite['writers'].str.split(',').apply(similarity_writers)

    return base_reduite

################################ ENTETE #######################################

image_path = images_path + "logo.png"
image = Image.open(image_path)
st.image(image)

st.header('Recommandation de films')

############################### RECHERCHE ######################################
st.subheader('')
st.subheader('Retrouver un film dans la base')

# Input
pattern = ''
pattern = st.text_input('Titre du film ou mot clé', '')

film_select = ''

if pattern != '':

    # Fonction de recherche_film 
    list_rech = recherche_film(base_movies, pattern)

    # Sélection du film dans la liste
    if len(list_rech) == 0:
          film_select = ''
          st.error("aucun film sélectionné")       
    elif len(list_rech) == 1:
        film_select = list_rech[0]
    else:
        film_select = st.selectbox(label = 'Sélectionner votre film dans la liste', index = 0, options = (list_rech))

if film_select != '':
    cle_IMDb = film_select[:9]
    fiche = fiche_film(base_movies, base_genres, cle_IMDb)
    for k,v in fiche.items():
        st.write (k,v)   
    # Image cliquable vers la fiche IMDb
    url_IMDb = "https://www.imdb.com/title/" + cle_IMDb
    balise = "<a href=" + url_IMDb + " target=""_blank""> <img src=""https://icons.iconarchive.com/icons/chrisbanks2/cold-fusion-hd/32/imdb-2-icon.png"" /> </a>"
    st.markdown(balise, unsafe_allow_html=True)

######################### RECOMMANDATION ###################################
    st.subheader('')
    st.subheader('Recommandation de films')
    
    # Réglage du nombre de propositions
    max_prop = st.slider('Combien de films souhaitez-vous obtenir ?', 1, 10, 5) + 1 # On ajoute 1 car le plue proche voisin est le film lui-même

    if st.button('Lancer la recommandation'):
        st.write ('Un peu de patience, je travaille...')
        st.write ('')

        # Définition du dataset à entrainer
        liste = base_movies['genres'].str.split(',')[base_movies['tconst'] == cle_IMDb].values    # Extrait les genres du film de référence
        base = full_base_reduite(base_movies, cle_IMDb)     # réduction de la base en fonction des critères du film (voir commentaires dans fonction)
        X = base[['runtimeMinutes_norm','scoreIMDB', 'similarity_d'] + liste[0]] # sélection des colonnes à utiliser pour l'algo

        # Entrainement du modèle
        distanceKNN = NearestNeighbors(n_neighbors = max_prop).fit(X) #calcul des n films les plus proches grace au modèle

        # Récupération des résultat dans un dataframe
        movie_kneighbors = distanceKNN.kneighbors(base.loc[base['tconst'] == cle_IMDb, X.columns])

        # Affichage des résultats
        for i in range(1,len(movie_kneighbors[1][0])):
            cle_IMDb = base.iloc[movie_kneighbors[1][0][i]][0]
            film_select = descriptif_film(base, cle_IMDb)
            st.write (film_select) 
            
            # Image cliquable vers la fiche IMDb
            url_IMDb = "https://www.imdb.com/title/" + cle_IMDb
            balise = "<a href=" + url_IMDb + " target=""_blank""> <img src=""https://icons.iconarchive.com/icons/chrisbanks2/cold-fusion-hd/32/imdb-2-icon.png"" /> </a>"
            st.markdown(balise, unsafe_allow_html=True)

# It's all folks