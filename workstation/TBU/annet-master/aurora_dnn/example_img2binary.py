BinaryFile = np.zeros((272543,901), dtype=np.uint8)

count = 0

for j in xrange(0,272543):

    #SOIL IMAGE ADJOINT
    b = temp[j,0,:,:]
    g = temp[j,1,:,:]
    r = temp[j,2,:,:]
    n = tempnir[j,:,:,:]

    g = g.reshape(1,225)
    r = r.reshape(1,225)
    b = b.reshape(1,225)
    n = n.reshape(1,225)

    BinaryFile[count,1:901] = np.concatenate( (r,g,b,n), axis=1 )
    #BinaryFile[count,1:962] = ndvi
    #BinaryFile[count,0]= 0
    BinaryFile[count,0]= indexes[count,1]
    #BinaryFile[count,1]= indexes[count,1]

    #print(BinaryFile[count,1])
    #print(BinaryFile[count,2])

    count = count +1

newFile = open ("yourfile.bin", "wb")
newFileByteArray = bytearray(BinaryFile)
newFile.write(newFileByteArray)