
import numpy as np
import pandas 
from matplotlib import pyplot
import tensorflow as tf


def getData():
    np.random.seed(1)
    # Data Extraction
    data = pandas.read_csv("/home/abhijeet/Documents/TornadoShit/sandp500/all_stocks_5yr.csv", delimiter=",")
    data = data[data["Name"] == "AAL"]

    x = np.array(data.drop(["Name","date", "open", "volume"], 1))
    np.random.shuffle(x)
    x = (x - np.mean(x))/np.std(x)

    y = np.array(data["open"])
    y = np.reshape(y, (y.shape[0], 1))
    np.random.shuffle(y)
    y = (y - np.mean(x))/np.std(y)

    return (x[0:(90*(len(x)//100))], x[(90*(len(x)//100)):], y[0:(90*(len(y)//100))], y[(90*(len(y)//100)):])



x, _, y, _ = getData()
print(x.shape, y.shape)


# In[23]:


def calc_cost(y_dash, y):
    m = y.shape[0]

    # print(np.sum(np.square(np.subtract(y_dash, y))))
    cost = (1/(2*m))*np.sum(np.square(np.subtract(y_dash, y)))
    cost = np.squeeze(cost)
    return cost


# In[24]:


def initialize_weights(input_layer, number_layers, n_nodes):
    
    W = {}
    b = {}
    temp_layer = input_layer
    
    for i in range(number_layers):
        W["W" + str(i)] = np.random.randn(n_nodes[i], temp_layer)
        b['b' + str(i)] = np.random.randn(n_nodes[i], 1)
        temp_layer = n_nodes[i]
        
    print(W["W0"].shape, b["b0"].shape)
    return W, b


# In[25]:


initialize_weights(3, 2, [4,1])


# In[26]:


def feed_forward(x_train, W, b):
    
#     variable to store previous outputs of the layers
    cache = {}

    z = x_train
#     For Layer 1
    for i in range(len(W)):
        cache["layer"+str(i)] = z
        z = np.dot(W["W" + str(i)], z.T)
        z = np.add(z, b["b" + str(i)])
        z = z.T
    
    cache["layer"+str(len(W))] = z

    return z, cache       


# In[27]:


def calculate_gradients(W, b , cache, y_train , number_layers, learning_rate):
    
#     For Layer number of layers - 1
    grads = {}
    m = y_train.shape[0]
    
    output = y_train
    
    for i in reversed(range(number_layers)):
        grads["dw" + str(i)] = learning_rate*(1/m)*np.sum(np.dot(np.subtract(cache["layer" + str( i + 1)],output).T,cache["layer" + str(i)]), axis = 1)
        
        grads["db" + str(i)] = learning_rate*(1/m)*np.sum(np.subtract(cache["layer" + str( i + 1)],output))
        
        if len(grads["dw" + str(i)].shape) == 1:
            grads["dw" + str(i)] = np.reshape( grads["dw" + str(i)], ( grads["dw" + str(i)].shape[0], 1))
        
        if len(grads["db" + str(i)].shape) == 1:
            grads["db" + str(i)] = np.reshape( grads["db" + str(i)], ( grads["db" + str(i)].shape[0], 1))
            
        output = cache["layer" + str( i + 1)]
    
    return grads


# In[28]:


def update_parameters(W, b, grads):
    
    for i in range(len(W)):
        W["W"+str(i)] -= grads["dw"+str(i)]
        b["b"+str(i)] -= grads["db"+str(i)]
        


# In[36]:


def calculate_accuracy(x_test, y_test, W , b):
    
#     print(x_test.shape)
#     print(y_test.shape)
#     print(W["W0"].shape)
#     print(W["W1"].shape)
    
    z = x_test
    
    for i in range(len(W)):
        
        z = np.dot(W["W" + str(i)], z.T)
        z = np.add(z, b["b" + str(i)])
        z = z.T
        print(z.shape)
    
        
    print(z[0:5])
    print(y_test[0:5])
    u = np.abs(y_test - z)
    print(u[0:5])
    

    u = np.divide(np.abs(u-y_test),y_test )*100
    u = np.mean(u)
    
    return u


# In[43]:


if __name__ == "__main__":

    np.random.seed(1)

    learning_rate = 0.005
    number_layers = 2 # 1 hidden and 1 output layers
    number_nodes = [4, 1]
    
    x_train, x_test, y_train, y_test = getData()
    cost_list = []

    # Parameter Initialization
    W, b = initialize_weights(x_train.shape[1], number_layers, number_nodes)

    cost_list = []
    # # Training Model
    for i in range(0,1):

        # Feed Forward z = w*x + b
        z, cache = feed_forward(x_train, W, b)
        cost = calc_cost(z, y_train)
        grad = calculate_gradients(W, b, cache, y_train, number_layers, learning_rate)
        update_parameters(W, b, grad)
        
        if i%10 == 0 :
#             print("Epoch : " , i )
            cost_list.append(cost)
            
#     pyplot.plot([i for i in range(len(cost_list))], cost_list)
#     pyplot.show()
    
    print("Accuracy : ", calculate_accuracy(x_test, y_test, W, b))
        
