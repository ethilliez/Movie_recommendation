import yaml
import logging

class config:
    def __init__(self, config_file):
        self.config_file = config_file

    def load(self):
        ''' Load JSON configuration file in `config_file'. The output is a
        dictionary with the same structure as in `config_file'. '''
        with open(self.config_file, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        return cfg

    def from_dict(self, cfg):
        ''' Parse the configuration from a dictionary format to the required variables.'''
        path = cfg['path-parameters']['path']
        file_plots  = cfg['path-parameters']['file_plots']
        file_movies = cfg['path-parameters']['file_movies']
        N_trees = int(cfg['annoy-parameters']['N_trees'])
        N_nodes = int(cfg['annoy-parameters']['N_nodes'])
        N_tops = int(cfg['annoy-parameters']['N_tops'])
        output_annoy_name = cfg['annoy-parameters']['output_annoy_name']
        chosen_movie = cfg['chosen_movie']
        return  path, file_plots, file_movies, N_trees, N_nodes, N_tops, output_annoy_name, chosen_movie 

    def parse(self):
        ''' Main function which loads and parse the configurable parameters into the required variables.'''
        # Load config
        cfg = self.load()
        # Parse to dictionnary
        path, file_plots, file_movies, N_trees, N_nodes, N_tops, output_annoy_name, chosen_movie = self.from_dict(cfg)
        return path, file_plots, file_movies, N_trees, N_nodes, N_tops, output_annoy_name, chosen_movie 

