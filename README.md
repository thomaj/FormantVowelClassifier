# FormantVowelClassifier

Classifies a formant (bands of sound energy) to the vowel sound that created it.  Overall,
there are 10 vowel sound classes.  The formants are points in 2-dimensional space,
so 2-dimensional gaussians are used.  It is assumed, for simplicity, that the models have
diagonal covariance.

## Files
  SimpleClassifier.py
  EMClassifier.py

### SimpleClassifier.py
  SimpleClassifier.py contains the script to perform the classification of a formant to a vowel.
  This uses the bayes net 'vowel --> formant'.

  ##### To Run:
    Be in the same directory as 2A.py and run the script with
    `python 2A.py [training_data] [testing_data]` where `[training_data]` is the training set and
    `[testing_data]` is the testing set for the classifier

### EMClassifier.py
  EMClassifier.py contains the script to perform the classification of a formant to a vowel.
  This uses the bayes net 'vowel --> mixture_vowel --> formant'.  The EM Algorithm is used to
  train this model and gaussians used for the mixture_vowel

  ##### To Run:
    Be in the same directory as 2B.py and run the script with
    `python EMClassifier.py [training_data] [testing_data] [num_gaussians]` where `[training_data]`
    is the training set, `[testing_data]` is the testing set for the classifier, and
    `[num_gaussians]` is the number of gaussians (classes) for each vowel in the mixture_vowel
    mixture of gaussians.