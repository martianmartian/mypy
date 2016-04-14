
================ numpy Tricks ================
'''fancy indexing'''
import numpy as np

     ---- randomly pick value on axis=1 from all axis=0 ----
    # values = np.random.random((100,5))
    values = np.arange(0,500).reshape(100,5)
    index_0 = range(values.shape[0])
    index_1 = np.random.randint(low=0,high=values.shape[1],size=(values.shape[0],))
    print values[index_0,index_1]

     ---- pick neurons by index within a column/cone... ----
     '# only vary within the limit of 1-5 range...'
    [  1   8  13  17  22  25  34  35  44  48  53  58  64  69  74  76  82  86
      93  96 104 109 113 119 120 127 134 136 140 148 152 159 161 166 170 177
     180 186 192 197 202 207 212 219 221 226 233 239 241 249 252 258 261 268
     271 279 284 289 294 295 303 307 314 318 320 329 331 336 344 345 350 358
     362 366 373 376 381 386 390 395 400 408 412 419 420 429 433 439 440 448
     452 458 461 469 473 478 482 486 494 499]
    [  1   6  13  16  24  29  32  39  41  47  52  57  61  65  73  79  81  88
      91  95 101 105 112 117 122 126 130 138 140 147 154 157 164 166 174 177
     184 185 194 199 200 208 214 217 221 225 233 237 240 245 251 257 262 266
     274 276 283 285 290 298 300 306 313 315 322 329 330 336 344 347 353 357
     361 367 370 375 383 389 392 395 402 405 410 417 424 429 434 438 444 445
     453 459 462 466 471 477 484 487 493 498]
    [  0   9  13  17  23  27  30  36  42  47  52  56  63  67  74  78  84  85
      90  97 104 106 114 119 123 126 132 136 142 147 152 158 163 168 173 178
     184 187 193 198 200 206 211 218 224 226 230 239 243 246 250 257 260 269
     271 276 281 289 294 295 304 308 313 315 320 325 330 336 344 349 354 357
     360 366 370 377 382 389 394 397 404 405 412 417 420 426 433 436 443 448
     451 458 463 467 471 477 484 488 492 497]


'''argsort inexing'''
# y = np.array([3, 1, 2, 5, 6, 4])
y = np.array(['A','D','C','G','B','F','E'])
x = np.random.random(y.shape) 
sortedIndex = np.argsort(x)
# print y[sortedIndex]
print y[np.argsort(x)]



''' create one more dimension '''
selected = selected[:, np.newaxis]
print selected.shape
print selected



'''thresholding / comparing  true/false vector'''
p=0.5
H1 = np.arange(10).reshape((2,5))
print H1
U1 = np.random.rand(*H1.shape)

print U1
U1 = U1<p
print U1
# U1 = np.random.rand(*H1.shape) < p   # Binary mask
H1 *= U1
print H1



print True+0.001
print False+0.001



'''high dimension vector reshape'''
X_train = np.random.random((100,5,5,2))
print X_train.shape
X_train = np.reshape(X_train, (X_train.shape[0], -1))
print X_train.shape


'''subsample data / extract'''
# X_train : (500000,32,32,3)
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]
