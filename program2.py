# CS 445, Summer 2021 - Programming Assignment 2 - Dan Jang
# Gaussian Na�ve Bayes classification

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy
# import matplotlib


class Program2(object):


    def __init__(thefile, name):
        thefile.trainingdata, thefile.testingdata, thefile.trainingset, thefile.testingset = thefile.splitfile(name)


    def splitfile(thefile, name):
        cache = numpy.loadtxt(name, delimiter=",")
        x = cache[:, :-1]
        y = cache[:, -1]
        trainingx, testingx, trainingy, testingy = train_test_split(x, y, test_size = 0.5)
    
        return trainingx, testingx, trainingy, testingy


    def standardmean(thefile):
        shape = thefile.trainingdata.shape[1]

        standard = numpy.ones((2, shape)) # 2-by Array for Good (not spam) and Bad (spam) data stuff
        themean = numpy.ones((2, shape))

        good = [] # Bi-dimnsional array for not spam-like data
        bad = [] # Bi-dimensional array for spammy data

        for idx in range(thefile.trainingdata.shape[0]):
            if thefile.trainingset[idx] == 0:
                good.append(thefile.trainingdata[idx])
            else:
                bad.append(thefile.trainingdata[idx])

        good = numpy.asarray(good)
        bad = numpy.asarray(bad)

        for idx in range(0,shape):
              themean[0,idx] = numpy.nanmean(good.T[idx])
              themean[1,idx] = numpy.nanmean(bad.T[idx])

              standard[0,idx] = numpy.std(good.T[idx])
              standard[1,idx] = numpy.std(bad.T[idx])

        for idx2 in range(shape):
            if standard[0,idx2] == 0:
                standard[0,idx2] = 0.0001
            if standard[1,idx2] == 0:
                standard[1,idx2] = 0.0001

        return themean,standard


    def probability(thefile):

        trainingbad = (numpy.count_nonzero(thefile.trainingset) / len(thefile.trainingset))
        traininggood = 1 - trainingbad

        trainingmeantuple, trainingstandardtuple = thefile.standardmean()
        return trainingbad, traininggood, trainingmeantuple, trainingstandardtuple


    def posterior(thefile, x, themean, standard, bad, good):

        posteriorprobability = numpy.ones(2)
        for idx in range(2):
            for idx in range(2):
                probability = 0

                if idx == 0:
                    probability += numpy.log(good)
                elif idx == 1:
                    probability += numpy.log(bad)

                for idx2 in range(len(x)):
                    a = ((x[idx2] - themean[idx][idx2]) ** 2)
                    b = (2 * ((standard[idx][idx2]) ** 2))

                    exp = numpy.exp(-1 * (a/b))
                    n = 1 / (numpy.sqrt(2 * numpy.pi) * standard[idx][idx2])

                    probability2 = (n * exp)
                
                    if probability2 == 0:
                        probability2 = (10 ** -320)

                    probability += numpy.log(probability2)

            posteriorprobability[idx] = probability

        return posteriorprobability


    def classification(thefile):

        badp, goodp, themean, standard = thefile.probability()

        classification = []

        for idx in range(thefile.testingdata.shape[0]):
            probability3 = thefile.posterior(thefile.testingdata[idx], themean, standard, badp, goodp)
            classification.append(numpy.argmax(probability3))

        classification2 = []
        classification2 = numpy.asarray(classification)

        return classification2


    def accuracy(thefile, classification):

        right = 0

        for idx in range(len(thefile.testingset)):
            if classification[idx]  == thefile.testingset[idx]:
                right += 1

        result = right / len(thefile.testingset)

        return result


    def precision(thefile, classification):

        rightprecision = 0
        wrongprecision = 0
        rightnum = 0
        wrongnum = 0

        for idx in range(len(thefile.testingset)):
            if classification[idx] == 1 and thefile.testingset[idx] == 1:
                rightprecision += 1
            elif classification[idx] == 0 and thefile.testingset[idx] == 0:
                rightnum += 1
            elif classification[idx] == 1 and thefile.testingset[idx] == 0:
                wrongprecision += 1
            else:
                wrongnum += 1

        return rightprecision, rightnum, wrongprecision, wrongnum


    def gaussian(thefile):

        gbayes = GaussianNB()

        gbayes.fit(thefile.trainingdata, thefile.trainingset)

        yprediction = gbayes.predict(thefile.testingdata)
        return accuracy_score(thefile.testingset, yprediction)

theprogram = Program2("spambase.data")

theprediction = theprogram.classification()

theaccuracy = theprogram.accuracy(theprediction)

rightp, rightn, wrongp, wrongn = theprogram.precision(theprediction)

theprecision = rightp / (rightp + wrongp)
therecall = rightp / (rightp + wrongn)
thematrix = confusion_matrix(theprogram.testingset, theprediction)

print("Confusion Matrix for the Bayes program: ")
print(thematrix)

print("Accuracy: ")
print(theaccuracy)
print("Comparative Accuracy: ")
print(theprogram.gaussian())

print("Precision: ")
print(theprecision)

print("Recollections: ")
print(therecall)

#Confusion Matrix for the Bayes program:
#[[1111  275]
# [ 271  644]]
#Accuracy:
#0.7627118644067796
#Comparative Accuracy:
#0.8183398522381573
#Precision:
#0.7007616974972797
#Recollections:
#0.7038251366120218

#Confusion Matrix for the Bayes program:
#[[1083  298]
# [ 209  711]]
#Accuracy:
#0.7796610169491526
#Comparative Accuracy:
#0.8126901347240331
#Precision:
#0.7046580773042617
#Recollections:
#0.7728260869565218

#Confusion Matrix for the Bayes program:
#[[1124  277]
# [ 223  677]]
#Accuracy:
#0.782703172533681
#Comparative Accuracy:
#0.8148631029986962
#Precision:
#0.709643605870021
#Recollections:
#0.7522222222222222