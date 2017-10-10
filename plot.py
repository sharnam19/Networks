from NN import NN

if __name__ == "__main__":
    model  = NN.load("model.json")
    model.plot()
