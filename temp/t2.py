import collections

counter = collections.Counter()
counter['a'] += 1
counter['a'] += 1
counter['b'] += 1
counter['b'] += 1

print(sum(counter.values()))