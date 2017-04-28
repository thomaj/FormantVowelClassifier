import sys
import numpy as np

# CSE 5522 Assignment
# Determining the vowel based on a formant's values using gaussian
# distributions and bayes rule.
#
# Author: Josh Thomas
# 4/12/17



# Reads in the data and returns a dictionary object with the data
def readInData(data_file):
    dataForVowels = {'ah': [], 'ao': [], 'ax': [], 'ay': [], 'eh': [], 'ey': [], 'ih': [], 'iy': [], 'ow': [], 'uw': []}
    for line in data_file:
        arr = line.split()
        dataForVowels[arr[2]].append([float(arr[0]), float(arr[1])])

    return dataForVowels

# Computes the gaussians for each vowel and the probability of each vowel
def getGaussiansAndProbs(data):
    gaussians = {
        'ah': {'numberOf': 0, 'X': [], 'Y': []},
        'ao': {'numberOf': 0, 'X': [], 'Y': []},
        'ax': {'numberOf': 0, 'X': [], 'Y': []},
        'ay': {'numberOf': 0, 'X': [], 'Y': []},
        'eh': {'numberOf': 0, 'X': [], 'Y': []},
        'ey': {'numberOf': 0, 'X': [], 'Y': []},
        'ih': {'numberOf': 0, 'X': [], 'Y': []},
        'iy': {'numberOf': 0, 'X': [], 'Y': []},
        'ow': {'numberOf': 0, 'X': [], 'Y': []},
        'uw': {'numberOf': 0, 'X': [], 'Y': []}
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
        gaussians[vowel]['X'] = [meanX, stddevX]
        gaussians[vowel]['Y'] = [meanY, stddevY]
        gaussians[vowel]['numberOf'] = len(formants)

        # Reset the totals
        totalX = totalY = 0
        deviationsX = deviationsY = 0

    # Have the counts of each vowel, so make the probabilities
    pOfVowel = []
    for vowel in gaussians.items():
        pOfVowel.append([vowel[0], float(vowel[1]['numberOf'])/totalNumOfEverything])

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


# Gets the condition probability and returns it based on the gaussian
def getCondProb(formant, gaussians2d):
    # Get the probability from the x axis
    mean = gaussians2d['X'][0]
    stddev = gaussians2d['X'][1]
    probX = gaussianFunction(mean, stddev, formant[0])

    # Get the probability from the y axis
    mean = gaussians2d['Y'][0]
    stddev = gaussians2d['Y'][1]
    probY = gaussianFunction(mean, stddev, formant[1])

    # Multiply them together because we assume they have diagonal covariance
    prob = probX * probY
    return prob


# Runs the test data through the model and returns an object with the results
def runTest(test_data, gaussians, pOfVowel):
    totalCorrect = numOfTests = 0
    retVal = {'total': [0, 0], 'vowels': {vowel: [0, 0] for vowel in
        gaussians.keys()}}

    # Loop trhough all the formants for each vowel
    for expectedVowel, formants in test_data.items():
        for formant in formants:
            pOfFGivenV = [[vowel, getCondProb(formant, gaussians[vowel])] for vowel in gaussians.keys()]

            # Multiply these conditional probability by the prob of the vowel
            pOfVGivenF = []
            for pofgv, pov in zip(pOfFGivenV, pOfVowel):
                vowel = pov[0]
                pOfVGivenF.append([vowel, pofgv[1] * pov[1]])

            # Determine the predicted vowel given this formant
            highestProb = -1
            for prob in pOfVGivenF:
                if prob[1] > highestProb:
                    highestProb = prob[1]
                    actualVowel = prob[0]

            ### print actualVowel, expectedVowel

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


def main():
    # Read from command line the training and testing files
    train_name = sys.argv[1]
    test_name = sys.argv[2]


    # Read the file testing file
    with open(train_name, 'r') as train_data_file:
        train_data = readInData(train_data_file)
        gaussians, pOfVowel = getGaussiansAndProbs(train_data)
        #print gaussians
        #print pOfVowel

    # Have the gaussians and probability for each vowel
    # Using bayes rule, can use P(V|F) = P(F|V)P(V) to determine which vowel
    # sound is most likely given the formant
    with open(test_name, 'r') as test_data_file:
        test_data = readInData(test_data_file)
        outcome = runTest(test_data, gaussians, pOfVowel)

        # Print the accuracies
        print 'Total Accuracy:  {0:.4}\n'.format(float(outcome['total'][0]) / outcome['total'][1])
        for counts in outcome['vowels'].items():
            print '\'{0}\' accuracy: {1:.2}'.format(counts[0], float(counts[1][0]) / counts[1][1])



if __name__ == '__main__':
    main()