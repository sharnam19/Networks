from distutils.core import setup

setup(
    name = 'networks',
    packages = ['networks','networks/layers','networks/layers/util/','networks/layers/descent/'],
    version = '0.3.1',  # Ideally should be same as your GitHub release tag varsion
    description = 'Allows to create NN models',
    author = 'Shharrnam Chhatpar',
    author_email = 'sharnam19.nc@gmail.com',
    url = 'https://github.com/sharnam19/NN-Library',
    download_url = 'https://github.com/sharnam19/NN-Library/archive/0.3.1.tar.gz',
    keywords = ['machine-learning', 'deep-learning','neural-networks','linear regression',
                'logistic regression'],
    classifiers = [],
)
