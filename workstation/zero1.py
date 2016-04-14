

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'


import numpy as np

import collections

# print collections.Counter(['a', 'b', 'c', 'a', 'b', 'b'])

# c = collections.Counter()
# # print 'Initial :', c
# c.update('abcdaab')
# # print 'Sequence:', c
# c.update('aaaaa')
# # print 'Sequence:', c


# for letter in 'abcde':
#     print '%s : %d' % (letter, c[letter])


# c = collections.Counter('extremely')
# c['z'] = 0
# print c
# # print c.elements()
# print list(c.elements())
# print list(c.values())


# c = collections.Counter('extremely')
# print c
# print list(c.elements())
# print c.most_common(2)
# print c.most_common(2)[0]
# print c.most_common(2)[0][0]  # the label...








# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# for y, cls in enumerate(classes):
#   print 'y:', y, 'cls:',cls
#   # y: 0 cls: plane

# classes = [('e', 3), ('m', 1)]
# for a, b in classes:
#   print a,b



# c = collections.Counter()
# with open('/usr/share/dict/words', 'rt') as f:
#     for line in f:
#         c.update(line.rstrip().lower())

# print 'Most common:'
# for letter, count in c.most_common(3):
#     print '%s: %7d' % (letter, count)