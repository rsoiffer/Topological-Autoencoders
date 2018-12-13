from scipy.stats import ttest_ind

file1 = 'results/tenc_4_cifar2.txt'
file2 = 'results/auto_4_cifar2.txt'


def read_file(file):
    with open(file) as f:
        return [float(x) for x in f.read().split(',')[:-1]]


values1 = read_file(file1)
values2 = read_file(file2)
value, pvalue = ttest_ind(values1, values2, equal_var=True)

print(values1, values2)
print(pvalue)
