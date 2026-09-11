import math
import random

wh = [(0.5, -0.3), (-0.4, 0.6), (0.9, 0.2), (-0.6, -0.7)]
bh = [0.1, -0.2, 0.05, 0.3]
wo = [0.7, -0.5, 0.3, 0.9]
bo = -0.1

#

def sigmoid(x):
    return 1/(math.exp(-x) + 1)

def derivative_sigmoid(x):
    return x * (1 - x)

def dLdivdaout(y, aout):
    return -(y - aout)

def forwardpass(wh, wo, bh, bo):
    z1 = [0, 0, 0, 0]
    input = random.choice([[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
    label = input.pop()

    for i in range(len(z1)):
        z1[i] = input[0] * wh[i][0] + input[1] * wh[i][1] + bh[i]

    a1 = [0, 0, 0, 0]

    for x in range(len(z1)):
        a1[x] = sigmoid(z1[x])

    z2 = 0
    for y in range(len(wo)):
        z2 += wo[y]*a1[y]
    z2 += bo
    a2  = sigmoid(z2)

    return z1, a1, z2, a2, label, input

learning_rate = 0.2
epochs = 20000
counter = 0
correct = 0
false = 0
for epoch in range(epochs):
    z1, a1, z2, a2, label, input = forwardpass(wh, wo, bh, bo)
    #print("label:", label, ", pred: ", round(a2), ", loss: ", round((label - a2)**2, 3))
    if counter == 100:
        counter = 0
        print(epoch, ": correct: ", correct, ", false: ", false)
        correct = 0
        false = 0
    counter += 1
    if label == round(a2):
        correct += 1
    else:
        false += 1

    
    delta_out = dLdivdaout(label, a2) * derivative_sigmoid(a2)
    bo = bo - learning_rate * delta_out
    dwo = [0, 0, 0, 0]
    for c in range(len(a1)):
        dwo[c] = a1[c] * delta_out 

    delta_h = [0, 0, 0, 0]
    for h in range(len(delta_h)):
        delta_h[h] = delta_out * wo[h] * derivative_sigmoid(a1[h])
    bh = [bh[i] - learning_rate * delta_h[i] for i in range(len(bh))]
    dwh = []
    for u in range(4):
        buffer = (delta_h[u]*input[0], delta_h[u]*input[1])
        dwh.append(buffer)

    new_wh = []

    for w in range(len(wh)):
        wo[w] = wo[w] - learning_rate * dwo[w]
        buffer_2 = (wh[w][0] - learning_rate * dwh[w][0], wh[w][1] - learning_rate * dwh[w][1])
        new_wh.append(buffer_2)

    wh  = new_wh
