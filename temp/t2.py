import collections

counter = collections.Counter()
counter['a'] += 1
counter['a'] += 1
counter['a'] += 1
counter['a'] += 1
counter['a'] += 1
counter['a'] += 1

print(len(counter), sum(counter.values()), len(counter) / sum(counter.values()))