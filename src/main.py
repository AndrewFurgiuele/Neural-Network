import mnist_loader
import network


#net settings
epochs = 1
batchSize = 10
lRate = 3.0

#net size
netLayers = [784,30,10]

#getting data
train_data, val_data, test_data = mnist_loader.load_data_wrapper()





net = network.Network(netLayers)
#net.SGD(train_data, epochs, batchSize, lRate, test_data = test_data )
net.save('./savedNet')


'''
print(type(net.weights))

print(len(net.weights))
for a in net.weights:
    print(type(a))
    print(len(a))
    for b in a:
        print(len(b))

    print('break') 
'''


'''
for a in net.biases:
    print(type(a))
    print(len(a))
    for b in a:
        #print(type(b))
        #print(len(b))        
        for c in b:
            print(c)    
        print('break1')

    print('break2')'''
