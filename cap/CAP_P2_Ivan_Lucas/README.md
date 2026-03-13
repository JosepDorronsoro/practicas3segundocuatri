Se entregan dos versiones de la práctica. Cada una de ellas está ejecutada no solo desde máquinas distintas, sino desde entornos diferentes. Nos parecía interesante encontrar las diferencias de ejecución entre un entorno de Kaggle (Iván) y un entorno local en una MSI i7-13620H con una gráfica Nvidia RTX 4060 (Lucas). Los resultados son distintos, pero no demasiado. 

Hay algunas diferencias de implementación, pero no son muy notables. La versión de Stencil 1D que usa memoria compartida está en el archivo _cap-p2-ivan-ej2.ipynb_, el resto de implementaciones son muy parecidas para el resto de algoritmos. 

De correjir una, corríjase las versiones de Iván, pues están más completas (ambas se han desarrollado entre ambos integrantes del grupo de forma conjunta, pero ejecutadas desde los entornos accesibles por cada uno). Las de Lucas, sin embargo, obtienen, en general, mejores resultados pues están ejecutadas desde mejor hardware tanto en CPU como en GPU. 

El programa más rápido para el procesado de imagen es el de Lucas, consiguiendo una tasa de 38.26 fps procesando imágenes de 8k, frente a los 7.12 fps de la versión de Iván, ejecutada desde Kaggle. 