import random

input = random.choice([[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]])

label = input.pop()
print(label, input)