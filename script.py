# Import libraries
import numpy as np
import pandas as pd
import codecademylib3

# Import data
dogs = pd.read_csv('dog_data.csv')

# Subset to just whippets, terriers, and pitbulls
dogs_wtp = dogs[dogs.breed.isin(['whippet', 'terrier', 'pitbull'])]

# Subset to just poodles and shihtzus
dogs_ps = dogs[dogs.breed.isin(['poodle', 'shihtzu'])]

# Inspect first five rows of the "dogs" dataframe
print(dogs.head())

# Store the is_rescue values for 'whippet's in a variable called whippet_rescue
whippet_rescue = dogs.is_rescue[dogs.breed == 'whippet']

# Calculate the number of whippet rescues into a variable called num_whippet_rescues
num_whippet_rescues = np.count_nonzero(whippet_rescue)

# Print out num_whippet_rescues
print(num_whippet_rescues)

# Save the total number of "whippet" in a variable called num_whippets
num_whippets = len(whippet_rescue)

# Print out num_whippets
print(num_whippets)

# Import module for binomial test
from scipy.stats import binom_test

# Run binomial test and save the p-value as pval
pval = binom_test(x=6, n=100, p=0.08)

# Print out pval
print(pval)

# Analyze the binomial test based on pval
print("Using a significance threshold of 0.05,the p-value of {} indicates that 8% of whippets are rescues\n".format(pval))

# Save the weights of "whippet" breed as wt_whippets
wt_whippets = dogs.weight[dogs.breed == 'whippet']

# Save the weights of "terrier" breed as wt_terriers
wt_terriers = dogs.weight[dogs.breed == 'terrier']

# Save the weights of "pitbull" breed as wt_pitbulls
wt_pitbulls = dogs.weight[dogs.breed == 'pitbull']

# Import module for ANOVA
from scipy.stats import f_oneway

# Run an ANOVA test and save p-value as pval
Fstat, pval = f_oneway(wt_whippets, wt_terriers, wt_pitbulls)

# Print out pval
print(pval)

# Analyze the ANOVA test Using a significance threshold of 0.05
print("Since the p-value of {} is less than 0.05, we can conclude that at least one pair of dog breeds have significantly different average weights.\n".format(pval))

# Import module for Tukey's Range Test
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Run Tukey's Range Test and save results as tukey_result
tukey_result = pairwise_tukeyhsd(dogs_wtp.weight, dogs_wtp.breed)

# Print out tukey_result
print(tukey_result)

# Analyze the Tukey's Range Test
print("Pairs of dog breeds with different average wights are:\n1. Pitbull and Terrier\n2. Terrier and Whippet\n")

# Create a contingency table of dog colors by breed (poodle vs. shihtzu). Save the table as Xtab
Xtab = pd.crosstab(dogs_ps.color, dogs_ps.breed)

# Print out Xtab
print(Xtab)

# Import module for Chi-Square test
from scipy.stats import chi2_contingency

# Run Chi-Square test and save the p-value as pval
chi2, pval, dof, expected = chi2_contingency(Xtab)

# Print out pval
print(pval)

# Analyze Chi-Square test using a significant threshold of 0.05
print("For p-value of {} < 0.05, we can conclude that there is not an association between breed (poodle vs. shihtzu) and color.\n".format(pval))

# Save the tail length of "rottweiler" breed as tl_rottweilers
tl_rottweilers = dogs.tail_length[dogs.breed == 'rottweiler']

# Save the tail length of "greyhound" breed as tl_greyhounds
tl_greyhounds = dogs.tail_length[dogs.breed == 'greyhound']

# Save the tail length of "chihuahua" breed as tl_chihuahuas
tl_chihuahuas = dogs.tail_length[dogs.breed == 'chihuahua']

# Run an ANOVA test and save p-value as pval
Fstat, pval = f_oneway(tl_rottweilers, tl_greyhounds, tl_chihuahuas)

# Print out pval
print(pval)

# Analyze the ANOVA test Using a significance threshold of 0.05
print("Since the p-value of {} is less than 0.05, we can conclude that at least one pair of dog breeds have significantly different average tail length.\n".format(pval))

# Subset to just rottweiler, greyhound, and chihuahua breeds
dogs_rgc = dogs[dogs.breed.isin(['rottweiler', 'greyhound', 'chihuahua'])]

# Run Tukey's Range Test and save results as tukey_results
tukey_results = pairwise_tukeyhsd(dogs_rgc.tail_length, dogs_rgc.breed)

# Print out tukey_results
print(tukey_results)

# Analyze the Tukey's Range Test
print("\nThe Tukey's Range Test shows that all the three breeds (rottweiler, greyhound and chihuahua) have different significantly average tail length.\n")

# Import module for boxplot
from matplotlib import pyplot as plt

# Find the three unique dog breeds (rottweiler, greyhound and chihuahua) and save it as breed_rgc
breed_rgc = dogs_rgc['breed'].unique()

# Create an empty list called datasets1
datasets1 = []

# Iterate through breed_rgc
for breed in breed_rgc:
  # Append the values of each breed's Average tail length to datasets1
  datasets1.append(dogs_rgc[dogs_rgc['breed'] == breed]['tail_length'].values)

# Create a large room for the plots
plt.figure(figsize=(7,4))

# Draw boxplots for datasets1
plt.boxplot(datasets1, labels = breed_rgc)

# Show the plots
plt.show()