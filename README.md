![alt text](http://www.kgncfm.com/wp-content/uploads/2016/10/images-7.jpg)

# Movie_recommendation

## Description:
This script is a movie recommendation engine which was built using a dataset of 21K movies from [CMU Movie Summary Corpus
](http://www.cs.cmu.edu/~ark/personas/). It firstly uses the Term Frequence Inverse Document Frequencies (TFIDF) to extract the main topic of each movie synopsis. It then uses the Spotify Annoy library which computes the cosinus distance between all synopsis and store them within an indexing tree. This indexing tree can then be queried for a recommendation based on a reference movie very quickly.

## Personal development goals:
- Practising implementing NLP algorithm such as tokenization, Stemming and TFIDF with the `nltk` Python library
- Practising building Annoy Tree index ([Spotify Library](https://github.com/spotify/annoy)) for fast queries

## Status of development:
- :white_check_mark: Data wrangling implemented
- :white_check_mark: TFIDF algorithm implemented
- :white_check_mark: Annoy tree indexing implemented
- :white_check_mark: Annoy tree querying implemented
- :white_check_mark: Documentation and config file implemented

## Requirements:
The main librairies required are: `nltk`, `pandas` and `annoy`. They can be installed using `pip install` or `conda install`.

## Execution:
1) Firsly, set the desired parameters in the `config.yaml` file such as paths, output names, chosen reference movie and Annoy tree parameters.
2) Execute the script via: `python3 main.py`

## Raw performance:
- By filtering and keeping a set of 13625 recent movies, encoding the TFIDF and building the Annoy tree index took 15 hours (only has to be done once). 
- Querying for a recommendation takes less than 1.09 seconds.
- Example of recommendations for the `Dark Knight` and `Home Alone` as movie references:
```
Reference: The Dark Knight 
Movie Name:  Attack of the 50 Ft. Woman - Distance:  1.3735449314117432
Movie Name:  Batman Begins - Distance:  1.3751866817474365
Movie Name:  Seeds of Arkham - Distance:  1.3760459423065186

Reference: Home Alone 
Movie Name:  Get Crazy - Distance:  1.3527811765670776
Movie Name:  A Christmas Carol - Distance:  1.3589770793914795
Movie Name:  Fun with Dick and Jane - Distance:  1.3603978157043457
```

Enjoy the movie.