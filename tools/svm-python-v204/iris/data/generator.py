from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

def store2file(filename, xs, ys):
    with open(filename, 'w') as f:
        for x, y in zip(xs, ys):
            f.write('{} '.format(y+1))
            for ind, ele in enumerate(x):
                f.write('{}:{} '.format(ind+1, ele))
            f.write('\n')

if __name__ == '__main__':
    d = load_iris()
    Xtr, Xte, Ytr, Yte = train_test_split(d.data, d.target)
    
    store2file('train.dat', Xtr, Ytr);
    store2file('test.dat', Xte, Yte);

