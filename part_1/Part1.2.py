import os
from surprise import Reader
from surprise import Dataset
from surprise import SVD, KNNBaseline
import time
from surprise.model_selection.search import GridSearchCV, RandomizedSearchCV

def getData(path, rating_scale):
    file_path = os.path.expanduser(path)
    reader = Reader(line_format='user item rating', sep=',', rating_scale=rating_scale, skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)
    return data

def al_SVD(data):
    start = time.time()
    param_grid = {
        'n_factors': [50, 100],
        'n_epochs': [10, 30],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.05, 0.1]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5, n_jobs=4)
    gs.fit(data)

    # best RMSE score
    print('Best score: ', gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    print('Best Configurations: ', gs.best_params['rmse'])

    end = time.time()
    print('Duration: ', end - start)

    # ---- Results-------
    # ratings_1.csv
    # Best score:  0.8861342999958881
    # Best Configurations:
    # {'n_factors': 100,
    #  'n_epochs': 30,
    #  'lr_all': 0.01,
    #  'reg_all': 0.1}
    # Duration: 389.934308052063

    # ratings_2.csv
    # Best score: 1.8349989677287641
    # Best Configurations:
    # {'n_factors': 100,
    #  'n_epochs': 30,
    #  'lr_all': 0.005,
    #  'reg_all': 0.1}
    # Duration: 31.674001216888428

def al_KNNBaseline(data):
    start = time.time()
    param_grid = {
        'k': [30, 40, 50],
        'min_k': [1, 5, 10],
        'sim_options': {
            'name': ['cosine', "msd", 'pearson', 'pearson_baseline'],
            'user_based': [True, False],
            'min_support': [3, 5],
            },
        'bsl_options': {
            'method': ['als'],
            'reg_i': [5, 10, 20],
            'reg_u': [5, 15, 20],
            'n_epochs': [10, 50, 100],
             }
    }
    gs = RandomizedSearchCV(KNNBaseline, param_grid, n_iter=20, measures=['rmse'], random_state = 1, cv=5, n_jobs=4)
    gs.fit(data)

    # best RMSE score
    print('Best score: ', gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    print('Best Configurations: ', gs.best_params['rmse'])
    end = time.time()
    print('Duration: ', end - start)

    # ---- Results-------
    # ratings_1.csv
    # Best score: 0.8865090682428004
    # Best Configurations:
    # {'k': 50,
    #  'min_k': 5,
    #  'sim_options': {'name': 'pearson_baseline',
    #                  'user_based': False,
    #                  'min_support': 3},
    #  'bsl_options': {'method': 'als',
    #                  'reg_i': 10,
    #                  'reg_u': 5,
    #                  'n_epochs': 50}}
    # duration: 631.4130008220673

    # ratings_2.csv
    # Best score: 1.8408340207641476
    # Best Configurations:
    # {'k': 30,
    #  'min_k': 1,
    #  'sim_options': {'name': 'pearson_baseline',
    #                  'user_based': True,
    #                  'min_support': 5},
    #   'bsl_options': {'method': 'als',
    #                   'reg_i': 20,
    #                   'reg_u': 20,
    #                   'n_epochs': 100}}
    # Duration: 40.6990008354187

def main():
    path = './dataset/ratings_1.csv'
    rating_scale = [1, 5]
    data = getData(path, rating_scale)

    print('KNNBaseLine ratings_1.csv')
    al_KNNBaseline(data)

    print('SVD ratings_1.csv')
    al_SVD(data)

    path = './dataset/ratings_2.csv'
    rating_scale = [1, 10]
    data = getData(path, rating_scale)

    print('KNNBaseLine ratings_2.csv')
    al_KNNBaseline(data)

    print('SVD ratings_2.csv')
    al_SVD(data)

if __name__ == '__main__':
    main()