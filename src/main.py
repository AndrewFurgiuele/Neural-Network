import mnist_loader
import network


#net settings
epochs = 5
batchSize = 10
lRate = 3.0

#net size
netLayers = [784,30,10]

#getting data
train_data, val_data, test_data = mnist_loader.load_data_wrapper()





net = network.Network(netLayers)
net2 = network.Network(netLayers)
#net.SGD(train_data, epochs, batchSize, lRate, test_data = test_data )
net.save('savedNet')
net2.load('savedNet')

net.SGD(train_data, epochs, batchSize, lRate, test_data = test_data )
net2.SGD(train_data, epochs, batchSize, lRate, test_data = test_data )

