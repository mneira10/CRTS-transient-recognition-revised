import numpy as np 
import matplotlib.pyplot as plt 

trainLoss = np.loadtxt('./trainLoss.dat',delimiter=' ')
plt.plot(trainLoss[:,0],trainLoss[:,1])
plt.xlabel('Epochs')
plt.ylabel('Cross entropy loss')
plt.savefig('trainLoss.png')
plt.close()


testMetrics = np.loadtxt('./testLoss.dat',delimiter=' ')

epochs = testMetrics[:,0]
precision = testMetrics[:,1]
recall = testMetrics[:,2]
fscore = testMetrics[:,3]

plt.plot(epochs,100*precision,label='Precision')
plt.plot(epochs,100*recall,label='Recall')
plt.plot(epochs,100*fscore,label='Fscore')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Percentage (%)')
plt.savefig('testMetrics.png')
