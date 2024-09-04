import numpy as np
mydata=np.array([[[1],[2],[3],[4],[5],[6],[7]],[[11],[22],[33],[44],[55],[66],[77]],[[111],[222],[333],[444],[555],[666],[777]]])
print(mydata.shape)
print(mydata.reshape(3,7).T)