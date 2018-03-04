from pandas import DataFrame
from xbos import XBOS

data = DataFrame(data={'attr1':[1,1,1,1,2,2,2,2,2,2,2,2,3,5,5,6,6,7,7,7,7,7,7,7,15],'attr2':[1,1,1,1,2,2,2,2,2,2,2,2,3,5,5,6,6,7,7,7,13,13,13,14,15]})
xbos = XBOS(n_clusters=3)
result = xbos.fit_predict(data)
for i in result:
    print(round(i,2))
