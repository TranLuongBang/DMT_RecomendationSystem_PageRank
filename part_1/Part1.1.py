import os
from surprise import Reader
from surprise import Dataset
from surprise import SVD, SVDpp, NMF, NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SlopeOne, CoClustering
from surprise.model_selection import KFold, cross_validate

def getData(path, rating_scale):
    file_path = os.path.expanduser(path)
    reader = Reader(line_format='user item rating', sep=',', rating_scale=rating_scale, skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)
    return data


def algorithms(data):
    kf = KFold(n_splits=5, random_state=0)

    print('----SVD----')
    cross_validate(SVD(), data, measures=['RMSE'], cv=kf, n_jobs=4,
                   verbose=True)

    print('----NMF----')
    cross_validate(NMF(), data, measures=['RMSE'], cv=kf, n_jobs=4,
                   verbose=True)

    print('-----Normal Predictor----')
    cross_validate(NormalPredictor(), data, measures=['RMSE'], cv=kf, n_jobs=4,
                   verbose=True)

    print('------BaselineOnly------')
    cross_validate(BaselineOnly(), data, measures=['RMSE'], cv=kf, n_jobs=4,
                   verbose=True)


    print('-----KNNBasic------')
    cross_validate(KNNBasic(), data, measures=['RMSE'], cv=kf, n_jobs=4,
                   verbose=True)

    print('-----KNNWithMeans----')
    cross_validate(KNNWithMeans(), data, measures=['RMSE'], cv=kf, n_jobs=4,
                   verbose=True)

    print('-----KNNWithZscore----')
    cross_validate(KNNWithZScore(), data, measures=['RMSE'], cv=kf, n_jobs=4,
                   verbose=True)

    print('-----KNNBaseLine----')
    cross_validate(KNNBaseline(), data, measures=['RMSE'], cv=kf, n_jobs=4,
                   verbose=True)

    print('-----SlopeOne-----')
    cross_validate(SlopeOne(), data, measures=['RMSE'], cv=kf, n_jobs=4,
                   verbose=True)

    print('-----CoClustering-----')
    cross_validate(CoClustering(), data, measures=['RMSE'], cv=kf, n_jobs=4,
                   verbose=True)

    print('----SVD++----')
    cross_validate(SVDpp(), data, measures=['RMSE'], cv=kf, n_jobs=4, verbose=True)

def main():
    path = './dataset/ratings_1.csv'
    rating_scale = [1, 5]
    data = getData(path, rating_scale)
    algorithms(data)

    path = './dataset/ratings_2.csv'
    rating_scale = [1, 10]
    data = getData(path, rating_scale)
    algorithms(data)





if __name__ == '__main__':
    main()
