import bdlb

fs = bdlb.load(benchmark="fishyscapes",download_and_prepare = False)
# automatically downloads the dataset
data = fs.get_dataset('Static')
print("hold")