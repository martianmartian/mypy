
================ numpy Tricks ================
'''fancy indexing'''
import numpy as np
scores = np.random.random((2,5))
a = [0,1,0]
b = [0,0,1]
print scores[a,b]

     ---- randomly pick value on axis=1 from all axis=0 ----
    # values = np.random.random((100,5))
    values = np.arange(0,500).reshape(100,5)
    index_0 = np.arange(values.shape[0])
    index_1 = np.random.randint(low=0,high=values.shape[1],size=(values.shape[0],))
    print values[index_0,index_1]

     ---- application: pick neurons by index within a column/cone... ----
     ---- application: pick pixels from an image ----
     '# only vary within the limit of 1-5 range...'
    [  '1'   8  13  17  22  25  34  35  44  48  53  58  64  69  74  76  82  86
      93  96 104 109 113 119 120 127 134 136 140 148 152 159 161 166 170 177
     180 186 192 197 202 207 212 219 221 226 233 239 241 249 252 258 261 268
     271 279 284 289 294 295 303 307 314 318 320 329 331 336 344 345 350 358
     362 366 373 376 381 386 390 395 400 408 412 419 420 429 433 439 440 448
     452 458 461 469 473 478 482 486 494 499]
    [  '3'   6  13  16  24  29  32  39  41  47  52  57  61  65  73  79  81  88
      91  95 101 105 112 117 122 126 130 138 140 147 154 157 164 166 174 177
     184 185 194 199 200 208 214 217 221 225 233 237 240 245 251 257 262 266
     274 276 283 285 290 298 300 306 313 315 322 329 330 336 344 347 353 357
     361 367 370 375 383 389 392 395 402 405 410 417 424 429 434 438 444 445
     453 459 462 466 471 477 484 487 493 498]
    [  '0'   9  13  17  23  27  30  36  42  47  52  56  63  67  74  78  84  85
      90  97 104 106 114 119 123 126 132 136 142 147 152 158 163 168 173 178
     184 187 193 198 200 206 211 218 224 226 230 239 243 246 250 257 260 269
     271 276 281 289 294 295 304 308 313 315 320 325 330 336 344 349 354 357
     360 366 370 377 382 389 394 397 404 405 412 417 420 426 433 436 443 448
     451 458 463 467 471 477 484 488 492 497]
    '''
    extract correct(100) neurons out of 500 (random for now)
    each sequential 5 neurons form a colume
    extraction can only be within each colume, one from each col
    '''
    neuronsVals = np.random.random(500)
    neuronsIndex = np.arange(0,500).reshape(100,5)
    ex_0 = np.arange(neuronsIndex.shape[0])
    ex_1 = np.random.randint(low=0,high=neuronsIndex.shape[1],size=(neuronsIndex.shape[0],))
    print neuronsVals[neuronsIndex[ex_0,ex_1]]



'''argsort inexing'''
        ---- sort labels based on respect values  ----

        all_labels = np.array(['A','D','H','C','I','G','B','F','E'])
        labels = all_labels[0:5]  #['A' 'D' 'H' 'C' 'I']
        values = np.random.randint(low=0,high=5,size=(labels.shape[0],))
        print values
        print np.argsort(values)
        print labels[np.argsort(values)]
        print all_labels[np.argsort(values)]




''' create one more dimension '''
        selected = selected[:, np.newaxis]
        print selected.shape
        print selected



'''thresholding / comparing  true/false vector'''
        p=0.5
        H1 = np.arange(10).reshape((2,5))
        U1 = np.random.rand(*H1.shape)
        H1 *= (U1<p)
        # H1 = H1*(U1<p)


        print True+0.001
        print False+0.001



'''high dimension vector reshape'''
        X_train = np.random.random((100,5,5,2))
        X_train = X_train.reshape(X_train.shape[0], -1)

        np.set_printoptions(suppress=False)
        img = np.random.randint(low=0,high=255,size=(3,32,32))
        # img = np.random.randint(low=0,high=255,size=(3,2,2))
        print img
        print "=================="
        print img.reshape(-1)
        print "=================="
        print img.reshape(3, -1)
        print "=================="


''' Subsample the data '''
        # X_train = np.random.random(50000)
        X_train = np.random.random((5000,32,32,3))
        # print X_train[0:500].shape
        num_training = 500
        mask = np.arange(num_training)
        X_train = X_train[mask]



================ collections / Counter Tricks ================
from collections import Counter
        
        'basics'
        # c = Counter(['a', 'b', 'c', 'a', 'b', 'b'])
        c = Counter()
        print 'Initial :', c
        c.update('abcdaab')
        print 'Sequence:', c
        c.update(['aa', 'b', 'c', 'a', 'b', 'b'])
        print 'Sequence:', c
        c.update('aaaaa')
        print 'Sequence:', c

        print c['a']
        for letter in 'abcde':
            print '%s : %d' % (letter, c[letter])


        c = Counter('extremelyfatthatamericaissfsdfsd')
        c['z'] = 0
        print list(c.elements())
        # ['e', 'e', 'e', 'm', 'l', 'r', 't', 'y', 'x']
        print list(c.values())
        # [3, 1, 1, 1, 1, 1, 1, 0]
        print c.most_common(2)
        print c.most_common(2)[0]
        print c.most_common(2)[0][0]  # the label...



        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for y, cls in enumerate(classes):
          print 'y:', y, 'cls:',cls
          # y: 0 cls: plane

        classes = [('e', 3), ('m', 1)]
        for a, b in classes:
          print a,b

        c = Counter('extremelyf')
        for letter in 'abcde':
            print '%s : %d' % (letter, c[letter])


        c = collections.Counter()
        with open('/usr/share/dict/words', 'rt') as f:
            for line in f:
                c.update(line.rstrip().lower())

        print 'Most common:'
        for letter, count in c.most_common(3):
            print '%s: %7d' % (letter, count)