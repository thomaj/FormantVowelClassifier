import sys
import random
import copy
import numpy as np

# CSE 5522 Assignment
# Determining the vowel based on a formant's values using gaussian
# distributions and bayes rule.
#
# Author: Josh Thomas
# 4/12/17

EPSILON = 0.01



# Reads in the data and returns a dictionary object with the data
def readInData(data_file):
    dataForVowels = {'ah': [], 'ao': [], 'ax': [], 'ay': [], 'eh': [], 'ey': [], 'ih': [], 'iy': [], 'ow': [], 'uw': []}
    for line in data_file:
        arr = line.split()
        dataForVowels[arr[2]].append([float(arr[0]), float(arr[1])])

    return dataForVowels


# Computes the gaussians for each vowel and the probability of each vowel
def getGaussiansAndProbs(data, numOfGaussians):
    prob = 1. / float(numOfGaussians)
    gaussians = {
        'ah': {'numOf': 0, 'gaussians': []},
        'ao': {'numOf': 0, 'gaussians': []},
        'ax': {'numOf': 0, 'gaussians': []},
        'ay': {'numOf': 0, 'gaussians': []},
        'eh': {'numOf': 0, 'gaussians': []},
        'ey': {'numOf': 0, 'gaussians': []},
        'ih': {'numOf': 0, 'gaussians': []},
        'iy': {'numOf': 0, 'gaussians': []},
        'ow': {'numOf': 0, 'gaussians': []},
        'uw': {'numOf': 0, 'gaussians': []}
    }

    totalX = totalY = 0
    deviationsX = deviationsY = 0
    totalNumOfEverything = 0
    # print data
    for vowel, formants in data.items():
        for numbers in formants:
            # print vowel
            # print numbers
            totalX = totalX + numbers[0]
            totalY = totalY + numbers[1]
            totalNumOfEverything = totalNumOfEverything + 1

        # Get the mean for both x and y gaussians for this vowel
        meanX = totalX / len(formants)
        meanY = totalY / len(formants)

        # Now we have the mean, so determine std. deviation
        ### deviationsX = [(num - meanX)**2 for num in formants]
        for numbers in formants:
            deviationsX = deviationsX + ((numbers[0] - meanX)**2)
            deviationsY = deviationsY + ((numbers[1] - meanY)**2)

        stddevX = np.sqrt([deviationsX / len(formants)])[0]
        stddevY = np.sqrt([deviationsY / len(formants)])[0]

        # Add these to the dictionary
        # Add a random number to initialize the two gaussians
        for i in range(numOfGaussians):
            randNum = random.uniform(-4, 4)
            newG = {'prob': prob, 'X': [meanX + randNum, stddevX + randNum], 'Y': [meanY + randNum, stddevY + randNum]}
            gaussians[vowel]['gaussians'].append(newG)

        gaussians[vowel]['numOf'] = len(formants)

        # Reset the totals
        totalX = totalY = 0
        deviationsX = deviationsY = 0

    # Have the counts of each vowel, so make the probabilities
    pOfVowel = []
    for vowel in gaussians.items():
        pOfVowel.append([vowel[0], float(vowel[1]['numOf'])/totalNumOfEverything])

    return gaussians, pOfVowel


# Computes the output of a gaussian distribution
def gaussianFunction(mean, stddev, input):
    exp = (input - mean)**2
    exp = exp / (2 * (stddev**2))
    exp = -exp

    base = np.e**exp

    divisor = 2 * np.pi
    divisor = np.sqrt([divisor])[0] * (stddev)

    return (1 / divisor) * base

# Calculates the standard deviation
def calcStdDev(sqXSum, mean, classWeight):
    rightSide = mean**2
    leftSide = sqXSum / classWeight
    return np.sqrt([leftSide - rightSide])[0]



########### EM Algorithm ################

# Performs the expectation step of the EM algorithm
# Determines the weighted probabilities that each formant belongs
# to that gaussian (class)
def expectation(formants, gaussians):
    allProbabilities = []
    i = 0
    #print gaussians
    # Loop through each formant and determine P(class|formant)
    for formant in formants:
        # Loop through each gaussian
        pOfCGivenX = []
        for gaussian in gaussians:
            #print gaussian
            # Probability on x axis
            mean = gaussian['X'][0]
            stddev = gaussian['X'][1]
            probX = gaussianFunction(mean, stddev, formant[0])
            # Probability on y axis
            mean = gaussian['X'][0]
            stddev = gaussian['X'][1]
            probY = gaussianFunction(mean, stddev, formant[0])

            prob = probX * probY
            # Calculate P(Class|X)
            pOfCGivenX.append(prob * gaussian['prob'])
            #print pOfCGivenX

        # Need to alpha normalize
        pOfCGivenX = pOfCGivenX / sum(pOfCGivenX)
        allProbabilities.append(pOfCGivenX)


    # Have the probability of each formant belonging to each gaussian
    # now sum up each probability for each gaussian to get that classes weight
    numGaussians = len(gaussians)
    weights = [0 for g in range(numGaussians)]
    for i in range(numGaussians):
        weights[i] = sum([classProb[i] for classProb in allProbabilities])

    return weights, allProbabilities

# Performs the maximization step of the EM algorithm
# Determines new mean, stddev, and probability of each gaussian (class)
def maximization(classWeights, formantWeights, formants, gaussians):
    # Loop through each gaussian class
    for gaussian, i in zip(gaussians, range(len(gaussians))):
        # Loop through each formant
        weightedTotalX = weightedTotalY = 0
        squaredWeightedTotalX = squaredWeightedTotalY = 0
        for weight, formant in zip(formantWeights, formants):
            # Add weighted x and y coordinate by gaussian i (0 or 1)
            weightedTotalX = weightedTotalX + formant[0] * weight[i]
            weightedTotalY = weightedTotalY + formant[1] * weight[i]

            # for std dev, need squared x weighted sum
            squaredWeightedTotalX = squaredWeightedTotalX + (formant[0]**2 * weight[i])
            squaredWeightedTotalY = squaredWeightedTotalY + (formant[1]**2 * weight[i])

        # Have gone through each formant, now calculate new mean
        gaussians[i]['X'][0] = weightedTotalX / classWeights[i]
        gaussians[i]['Y'][0] = weightedTotalY / classWeights[i]

        # Calculate new std dev
        gaussians[i]['X'][1] = calcStdDev(squaredWeightedTotalX, gaussians[i]['X'][0], classWeights[i])
        gaussians[i]['Y'][1] = calcStdDev(squaredWeightedTotalY, gaussians[i]['Y'][0], classWeights[i])

        # Change the probability of this gaussian P(Class)
        gaussians[i]['prob'] = classWeights[i] / sum(classWeights)

        #print gaussians


# Checks convergence
# Only looks at the means of each gaussian
def hasConverged(startG, nextG):
    # Check and see if each gaussian change is less than EPSILON
    for g1, g2 in zip(startG, nextG):
        if g1['X'][0] - g2['X'][0] > EPSILON:
            return False
        if g1['Y'][0] - g2['Y'][0] > EPSILON:
            return False
        if g1['X'][1] - g2['X'][1] > EPSILON:
            return False
        if g1['Y'][1] - g2['Y'][1] > EPSILON:
            return False

    return True   # Every gaussian has converged


# Performs the EM algorithm on one vowel's data
def EMAlgorithm(formants, gaussianData):
    #print gaussianData['gaussians']
    start =  copy.deepcopy(gaussianData['gaussians'])  # retain before values
    classWeights, formantWeights = expectation(formants, gaussianData['gaussians'])
    maximization(classWeights, formantWeights, formants, gaussianData['gaussians'])
    nextTimeStep = gaussianData['gaussians']

    # Continue the steps until convergence
    while(not hasConverged(start, nextTimeStep)):
        start =  copy.deepcopy(gaussianData['gaussians'])
        classWeights, formantWeights = expectation(formants, gaussianData['gaussians'])
        maximization(classWeights, formantWeights, formants, gaussianData['gaussians'])
        nextTimeStep = gaussianData['gaussians']


# Runs the EM algorithm on all of the vowels where each vowel has multiple
# gaussians to train
def runTraining(training_data, allGaussians):
    # Do EM algorithm for 1 vowel before moving on to do EM on the next vowel
    for vowel, formants in training_data.items():
        print 'Learning -- ' + str(vowel)
        EMAlgorithm(formants, allGaussians[vowel])



def getProbOfEachVowel(formant, gaussians, pOfV):
    vowelProbs = []
    for prob in pOfV:
        vowel = prob[0]
        sumOfModels = 0

        # Loop through each model
        for g in gaussians[vowel]['gaussians']:
            pOfFGivenM_X = gaussianFunction(g['X'][0], g['X'][1], formant[0])
            pOfFGivenM_Y = gaussianFunction(g['Y'][0], g['Y'][1], formant[1])
            pOfFGivenM = pOfFGivenM_X * pOfFGivenM_Y

            # Now calculate the joint with this model and add it to the sum
            rightSide = pOfFGivenM * g['prob'] * prob[1]
            sumOfModels = sumOfModels + rightSide

        # Add this probability to all of the probabilities
        vowelProbs.append([vowel, sumOfModels])

    # print vowelProbs
    return vowelProbs


def runTesting(testing_data, allGaussians, pOfV):
    totalCorrect = numOfTests = 0
    retVal = {'total': [0, 0], 'vowels': {vowel: [0, 0] for vowel in
        allGaussians.keys()}}

    # Loop through each formant in each vowel
    for expectedVowel, formants in testing_data.items():
        for formant in formants:
            pOfVAndF = getProbOfEachVowel(formant, allGaussians, pOfV)
            # print pOfVAndF

            # Determine the predicted vowel given this formant
            highestProb = -1
            for prob in pOfVAndF:
                if prob[1] > highestProb:
                    highestProb = prob[1]
                    actualVowel = prob[0]

            # Check to see if this is the correct vowel
            if actualVowel == expectedVowel:
                totalCorrect = totalCorrect + 1
                retVal['vowels'][actualVowel][0] = retVal['vowels'][actualVowel][0] + 1
            # Update total counts
            numOfTests = numOfTests + 1
            retVal['vowels'][expectedVowel][1] = retVal['vowels'][expectedVowel][1] + 1

    # set the return objects values to the correct counts
    retVal['total'][0] = totalCorrect
    retVal['total'][1] = numOfTests

    return retVal



# Main method to run
def main():
    # Read from command line the training and testing files
    train_name = sys.argv[1]
    test_name = sys.argv[2]
    number_of_models = 2


    # Read the file training file and train
    with open(train_name, 'r') as train_data_file:
        train_data = readInData(train_data_file)
        gaussianData, pOfVowel = getGaussiansAndProbs(train_data, number_of_models)
        #print gaussianData
        #print pOfVowel
        runTraining(train_data, gaussianData)

    # Now test
    with open(test_name, 'r') as test_data_file:
        print '\nBegin Testing'
        test_data = readInData(test_data_file)
        outcome = runTesting(test_data, gaussianData, pOfVowel)

        # Print the accuracies
        print 'Total Accuracy:  {0:.4}\n'.format(float(outcome['total'][0]) / outcome['total'][1])
        for counts in outcome['vowels'].items():
            print '\'{0}\' accuracy: {1:.2}'.format(counts[0], float(counts[1][0]) / counts[1][1])





if __name__ == '__main__':
    main()