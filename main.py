import nltk
import pandas as pd 
import os.path
import codecs
import logging
from config import config
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
        '''Read the movies file and plots file into separate dataframes and merge them.'''
        # Read the Movies file
        DF_infos = pd.read_table(self.path+self.file_movies, sep='\t')
        DF_infos.columns = ['Wikipedia_ID', 'Freebase_ID','Movie_name', 'Movie_release_date',
            'Movie_revenue','Movie_runtime','Movie_languages','Movie_countries','Movie_genres']
        DF_infos['Movie_genres']=DF_infos['Movie_genres'].apply(lambda x: re.findall(r': "(.*?)",',x))
        # Read the plots file
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
        # Merge both
        DF = pd.merge(DF_infos, DF_plot, on = 'Wikipedia_ID')
        # Perform some text standardization
        DF['Synopsis'] = DF['Synopsis'].str.lower()
        for char in ["\'",'*','[','(','-','"']:
            DF['Synopsis']  = DF['Synopsis'].replace(char,"")
        return DF

    def StemTokens(self, tokens):
        '''Apply PorterStemmer on tokens to enhance efficiancy of tokenization'''
        stemmer = nltk.stem.porter.PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

    def tf_all_movies(self, DF):
        '''Get the term frequency over all synopsis from the dataframe'''
        all_words_synopsis = [DF.iloc[i]['Synopsis'] for i in range(0,len(DF.index))]
        all_words_synopsis = ''.join(all_words_synopsis)
        tokens = nltk.word_tokenize(all_words_synopsis)
        tokens = self.StemTokens(tokens)
        fd = nltk.FreqDist(tokens)
        return fd

    def tfidf_matrix_movies(self, DF):
        '''Build the TDIDF matrix over all movies from the dataframe DF. '''
        # Get the term frequence of all words of all synopsis
        fd_all = self.tf_all_movies(DF)
        # Create list of all possible name from library
        name_list = [name for name in names.words('male.txt')] + [name for name in names.words('female.txt')]
        # For each synopsis
        matrix = []
        for i in range(0, len(DF.index)):
            row = []
            # Get term frequency description of the synopsis
            text = DF.iloc[i]['Synopsis']
            tokens = nltk.word_tokenize(text)
            tokens = self.StemTokens(tokens)
            fd_synopsis = nltk.FreqDist(tokens)
            # For each term over all movie synopsis
            for term in fd_all:
                # If this term in within this synopsis and not a name and an alphabetical string
                if(term.isalpha() and term not in name_list and term in fd_synopsis.keys()):
                    # Get and save inverse frequency of the term over all movies
                    tfidf_val = round(fd_synopsis[term]/fd_all[term],2)
                    row.append(tfidf_val)
                else:
                # If not, append 0 in the tdidf matrix row
                    row.append(0.00)
            matrix.append(row)
        return matrix 

    def build_indexing_tree(self, tdidf_matrix):
        '''Build an Annoy tree index of the TDIDF matrix'''
        # Initialize the Annoy tree with the cosinus distance
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
        '''Load the Annoy tree index from file.'''
        indexing_tree = glob.glob(self.path+"*.annoy")
        N_vector = indexing_tree[0].replace(self.path+self.output_annoy_name,'')
        N_vector = int(N_vector.replace('.annoy',''))
        annoy_index = annoy.AnnoyIndex(N_vector)
        annoy_index.load(indexing_tree[0])
        return annoy_index

    def query_indexing_tree(self, annoy_index,  DF, chosen_movie):
        '''Query the chosen movie in the annoy_index by getting its id in the DF dataframe.'''
        # Find index of movie reference in DF
        list_index = DF.index[DF.Movie_name.str.contains(chosen_movie)]
        # Found the N closest to the chosen movie
        Top_closest = annoy_index.get_nns_by_item(list_index[0], self.N_tops, 
                    search_k = self.N_nodes, include_distances=True)
        return Top_closest

    def main(self, chosen_movie):
        # Check if Annoy movie index exists
        index_file = glob.glob(self.path+"*.annoy")
        if(len(index_file) == 0):
            # If not, first read data 
            logger.info(" Reading data...")
            if(os.path.isfile(self.path+'Dataset.csv')):
                DF = pd.read_csv(self.path+'Dataset.csv')
            else:
                DF = self.read_data()
            DF.index = range(0,len(DF))
            logger.info(DF.head(5))
            DF.to_csv(self.path+'Dataset.csv')
            logger.info((" Number of movies: ", len(DF.index)))
            # Then build the TDIDF matrix over all movies
            logger.info(" Building the movies TDIDF matrix...")
            tdidf_matrix = self.tfidf_matrix_movies(DF)
            # Finaly build the Annoy tree index
            logger.info(" Building the indexing tree...")
            self.build_indexing_tree(tdidf_matrix)
        else:
            DF = pd.read_csv(self.path+'Dataset.csv')
        # Load the built Annoy index
        logger.info(" Loading Annoy index...")
        annoy_index = self.load_index()
        # Query for a movie similar to chosen movie
        logger.info(" Performing matching query...")
        Top_closest = self.query_indexing_tree(annoy_index, DF, chosen_movie)
        # Return the best matches
        Best_choices = {}
        for i in range(self.N_tops):
            Best_choices[Top_closest[1][i]]= DF.iloc[Top_closest[0][i]]['Movie_name']
        return Best_choices


if __name__ == '__main__':
    # Define the main parameters
    path, file_plots, file_movies, N_trees, N_nodes, N_tops, output_annoy_name, chosen_movie = config('config.yaml').parse()
    # Initialize the process
    process = movie_recommendation_engine(path, file_plots, file_movies, N_trees, N_nodes, N_tops, output_annoy_name)
    # Run the process
    Best_choices = process.main(chosen_movie)
    for key in Best_choices.keys():
        print("Movie Name: ", Best_choices[key],"- Distance: ",key)
