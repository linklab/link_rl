import collections

# initializing deque
de = collections.deque([0, 1, 2], maxlen=3)
print(de)

print(de[0])
print(de[1])
print(de[2])
print()

de.append(3)
print(de[0])
print(de[1])
print(de[2])