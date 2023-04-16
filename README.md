Dataset can be found - https://www.kaggle.com/datasets/salmansajid05/oral-diseases

Here is Dental Caries Prediction System, which is based on unsupervised machine learning algorithm K-means Clustering. This algorithm mainly used for clustering data points into different clusters. 


There libraries includes cv2, numpy, os, filedialog from tkinter, Kmeans from sklearn.cluster, metrics from sklearn, pyplot from matplotlib, and ImageTk, Image from PIL. It uses KMeans clustering to classify the input image as healthy or decayed based on its features. The system provides a graphical user interface (GUI) that allows the user to select an image. When the code is run, the tkinter window will be displayed from which user have to upload a test image. The model is trained using dataset provided in the path. 


The KMeans algorithm is applied to the X_train data with the number of clusters set to 2. The model is trained on the training data, and the predicted labels are stored in the y_pred list. 


The predicted label will be displayed on the Tkinter window as either "Healthy" or "Caries Found!" depending on the predicted label.


If the predicted class is 1, the function displays a message indicating that the tooth is healthy. If the predicted class is 0, the function displays a message indicating that caries have been found.
