# CS 445, Summer 2021 - Programming Assignment 2 - Dan Jang
# Gaussian Naive Bayes classification

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy
# import matplotlib


class Program2(object):


    def __init__(thefile, name):
        
        print("The program will run itself four separate times with each run having a 'new' dataset split!")
        print("")
        print("")
        percenter = float(100)
        for idxx in "four": # cycles 4 times
            thefile.trainingdata, thefile.testingdata, thefile.trainingset, thefile.testingset = thefile.splitfile(name)

            theprediction = thefile.classification()    

            theaccuracy = thefile.accuracy(theprediction)

            rightp, rightn, wrongp, wrongn = thefile.precision(theprediction)

            theprecision = rightp / (rightp + wrongp)
            therecall = rightp / (rightp + wrongn)
            thematrix = confusion_matrix(thefile.testingset, theprediction)

            if idxx == "f":
                print("1st Run - Confusion Matrix for the Bayes program: ")
            elif idxx == "o":
                print("2nd Run - Confusion Matrix for the Bayes program: ")
            elif idxx == "u":
                print("3rd Run - Confusion Matrix for the Bayes program: ")
            elif idxx == "r":
                print("4th Run - Confusion Matrix for the Bayes program: ")

            print(thematrix)

            print("Accuracy: ")
            print(theaccuracy)
            print("[" + (str((theaccuracy) * percenter)) + "%]")
            print("")

            print("Comparative Accuracy: ")
            print(thefile.gaussian())
            print("[" + (str((thefile.gaussian()) * percenter)) + "%]")
            print("")

            print("Precision: ")
            print(theprecision)
            print("[" + (str((theprecision) * percenter)) + "%]")
            print("")

            print("Recollections: ")
            print(therecall)
            print("[" + (str((therecall) * percenter)) + "%]")
            print("")
            print("")
            
            print("---")

            print("")
            print("")

    # This function would help split the Spambase database to create a testing set and a training set;
    # would supposedly contain around 40% spam and 60% non-spam to represent actual composition of the Spambase dataset per split file, each split having around 2300 instances
    def splitfile(thefile, name):
        cache = numpy.loadtxt(name, delimiter=",")
        x = cache[:, :-1]
        y = cache[:, -1]
        trainingx, testingx, trainingy, testingy = train_test_split(x, y, test_size = 0.5)
    
        return trainingx, testingx, trainingy, testingy

    # Helps to calculate standard deviation and mean values for each of the non-spam (good) and spam (bad) data
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

        # Calculates the mean value and standard deviation values  for specific types of our two types of arrays
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

    #  Generates our probability based prediction model
    def probability(thefile):

        trainingbad = (numpy.count_nonzero(thefile.trainingset) / len(thefile.trainingset))
        traininggood = 1 - trainingbad

        trainingmeantuple, trainingstandardtuple = thefile.standardmean()
        return trainingbad, traininggood, trainingmeantuple, trainingstandardtuple


    def posterior(thefile, x, themean, standard, bad, good):

        posteriorprobability = numpy.ones(2)
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

    # Uses the algorithmic Naive Bayes model as to classify
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

    # Benchmark comparison purposes, a Gaussian Naive Bayes implementation from sklearn.naive_bayes, 
    # that is ran with the same data set as the 'homebrew' implementation of GNB for our Program 2
    def gaussian(thefile):

        gbayes = GaussianNB()

        gbayes.fit(thefile.trainingdata, thefile.trainingset)

        yprediction = gbayes.predict(thefile.testingdata)
        return accuracy_score(thefile.testingset, yprediction)

theprogram = Program2("spambase.data")

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

# after fix (there was a double  for loop)
#Confusion Matrix for the Bayes program:
#[[1022  377]
# [  39  863]]
#Accuracy:
#0.8192090395480226
#Comparative Accuracy:
#0.8218166014776185
#Precision:
#0.6959677419354838
#Recollections:
#0.9567627494456763

