import pickle

k = open("/home/xelese/CapstoneProject/metadata/dump_bl-20190303-191514-0.pkl","rb")
p = pickle.load(k)

print(p)