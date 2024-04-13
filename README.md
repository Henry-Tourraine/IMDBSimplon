# IMDBSimplon
Pour lancer l'application, exécuter depuis el dossier racine :
```bat
docker build -t imdb .
docker run -it -p 127.0.0.1:4000:80 -e MODEL=rf imdb
```

puis rendez-vous sur (localhost)[http://localhost:4000/score]

Sélectionner un film pour obtenir la prédiction d'un avis.

La prédiction se fait à partir de l'ensemble des colonnes sauf :
- "cast_total_fb_likes"
- "plot_keywords"
- "movie_imdb_link"
- "gross"
- "movie_fb_likes"

L'algorithme Random forest est "fit" une première fois puis il est enregistré avec pickle et rechargé à chaque fois.