import nltk
import pandas as pd 
import os.path
import codecs
import logging
import math
import annoy
import glob
import re
from nltk.corpus import names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class movie_recommendation_engine:
    def __init__(self, path, file_plots, file_movies, N_trees, N_nodes, N_tops, output_annoy_name):
        self.path = path
        self.file_plots = file_plots
        self.file_movies = file_movies
        self.N_trees = N_trees
        self.N_nodes = N_nodes
        self.N_tops = N_tops
        self.output_annoy_name = output_annoy_name

    def read_data(self):
        DF_infos = pd.read_table(self.path+self.file_movies, sep='\t')
        DF_infos.columns = ['Wikipedia_ID', 'Freebase_ID','Movie_name', 'Movie_release_date',
            'Movie_revenue','Movie_runtime','Movie_languages','Movie_countries','Movie_genres']
        DF_infos['Movie_genres']=DF_infos['Movie_genres'].apply(lambda x: re.findall(r': "(.*?)",',x))
        #
        f = open(self.path+self.file_plots, 'r')
        inside = False
        plot = ''
        data = []
        for line in f:
            if line[0].isdigit() and not inside:
                inside = True
                Wikipedia_ID = int(line.split()[0].strip())
                plot = ' '.join(line.split()[1:])
            elif line[0].isdigit() and inside:
                data.append((Wikipedia_ID, plot))
                plot = ''
                inside = False
            else:
                plot += line.replace("\n", "")
        f.close()
        DF_plot = pd.DataFrame(data, columns = ['Wikipedia_ID', 'Synopsis'])
        DF = pd.merge(DF_infos, DF_plot, on = 'Wikipedia_ID')
        DF['Synopsis'] = DF['Synopsis'].str.lower()
        for char in ["\'",'*','[','(','-','"']:
            DF['Synopsis']  = DF['Synopsis'].replace(char,"")
        return DF

    def StemTokens(self, tokens):
        stemmer = nltk.stem.porter.PorterStemmer()
        return [stemmer.stem(token) for token in tokens]


    def tf_all_movies(self, DF):
        '''Get the term frequency over all synopsis'''
        all_words_synopsis = [DF.iloc[i]['Synopsis'] for i in range(0,len(DF.index))]
        all_words_synopsis = ''.join(all_words_synopsis)
        tokens = nltk.word_tokenize(all_words_synopsis)
        tokens = self.StemTokens(tokens)
        # Get pseudo inverse frequency of the term over all movies
        fd = nltk.FreqDist(tokens)
        return fd

    def tfidf_matrix_movies(self, DF):
        # Get the term frequence of all words of all synopsis
        fd_all = self.tf_all_movies(DF)
        # Create list of all possible name from library
        name_list = [name for name in names.words('male.txt')] + [name for name in names.words('female.txt')]
        # For each synopsis
        matrix = []
        for i in range(0, len(DF.index)):
            row = []
            # Term frequency description of the synopsis
            text = DF.iloc[i]['Synopsis']
            tokens = nltk.word_tokenize(text)
            tokens = self.StemTokens(tokens)
            fd_synopsis = nltk.FreqDist(tokens)
            # For each term over all movie synopsis
            for term in fd_all:
                if(term.isalpha() and term not in name_list and term in fd_synopsis.keys()):
                    # Get inverse frequency of the term over all movies
                    tfidf_val = round(fd_synopsis[term]/fd_all[term],2)
                    row.append(tfidf_val)
                else:
                    row.append(0.00)
            matrix.append(row)
        return matrix 

    def build_indexing_tree(self, tdidf_matrix):
        N_vector = len(tdidf_matrix[0])
        a = annoy.AnnoyIndex(N_vector, metric="angular")
        # Feed Annoy index
        for i in range(0, len(tdidf_matrix)):
            a.add_item(i,tdidf_matrix[i])
        # Build Annoy tree
        logger.debug(" Build the Annoy indexing tree")
        a.build(self.N_trees)
        # Save Annoy tree
        logger.debug(" Save the Annoy indexing")
        a.save(self.path+self.output_annoy_name+str(N_vector)+'.annoy')

    def load_index(self):
        indexing_tree = glob.glob(self.path+"*.annoy")
        N_vector = indexing_tree[0].replace(self.path+self.output_annoy_name,'')
        N_vector = int(N_vector.replace('.annoy',''))
        annoy_index = annoy.AnnoyIndex(N_vector)
        annoy_index.load(indexing_tree[0])
        return annoy_index


    def query_indexing_tree(self, annoy_index,  DF, chosen_movie):
        # Find index of movie reference
        list_index = DF.index[DF.Movie_name.str.contains(chosen_movie)]
        # Found the N closest to ref vector
        Top_closest = annoy_index.get_nns_by_item(list_index[0], self.N_tops, 
                    search_k = self.N_nodes, include_distances=True)
        return Top_closest

    def main(self, chosen_movie):
        index_file = glob.glob(self.path+"*.annoy")
        if(len(index_file) == 0):
        # Perform topic modeling and index building for all movies
            logger.info(" Reading data...")
            if(os.path.isfile(self.path+'Dataset.csv')):
                DF = pd.read_csv(self.path+'Dataset.csv')
            else:
                DF = self.read_data()
            DF.index = range(0,len(DF))
            print(DF.head(5))
            DF.to_csv(self.path+'Dataset.csv')
            logger.info((" Number of movies: ", len(DF.index)))
            logger.info(" Building the movies TDIDF matrix...")
            tdidf_matrix = self.tfidf_matrix_movies(DF)
            logger.info(" Building the indexing tree...")
            self.build_indexing_tree(tdidf_matrix)
        else:
            DF = pd.read_csv(self.path+'Dataset.csv')
        # Query for a movie similar to chosen movie
        logger.info(" Loading Annoy index...")
        annoy_index = self.load_index()
        logger.info(" Performing matching query...")
        Top_closest = self.query_indexing_tree(annoy_index, DF, chosen_movie)
        Best_choices = {}
        for i in range(self.N_tops):
            Best_choices[Top_closest[1][i]]= DF.iloc[Top_closest[0][i]]['Movie_name']
        return Best_choices

if __name__ == '__main__':
    # Define the main parameters
    path = 'Data/'
    file_plots = 'plot_summaries.txt' 
    file_movies = 'movie.metadata.tsv'
    N_trees = 10
    N_nodes = 50
    N_tops = 5
    output_annoy_name = 'movie_indexing'
    chosen_movie = 'Squibs'#'Mary Poppins'
    # Initialize the process
    process = movie_recommendation_engine(path, file_plots, file_movies, N_trees, N_nodes, N_tops, output_annoy_name)
    Best_choices = process.main(chosen_movie)
    for key in Best_choices.keys():
        print("Movie Name: ", Best_choices[key],"- Distance: ",key)
