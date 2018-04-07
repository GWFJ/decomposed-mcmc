# !/usr/bin/python


# MCMC Random Walk for Feedforward Neural Network for One-Step-Ahead Chaotic Time Series Prediction

# Data (Sunspot and Lazer). Taken' Theorem used for Data Reconstruction (Dimension = 4, Timelag = 2).
# Data procesing file is included.

# RMSE (Root Mean Squared Error)

# based on: https://github.com/rohitash-chandra/FNN_TimeSeries
# based on: https://github.com/rohitash-chandra/mcmc-randomwalk


# Rohitash Chandra, Centre for Translational Data Science
# University of Sydey, Sydney NSW, Australia.  2017 c.rohitash@gmail.conm
# https://www.researchgate.net/profile/Rohitash_Chandra



# Reference for publication for this code
# [Chandra_ICONIP2017] R. Chandra, L. Azizi, S. Cripps, 'Bayesian neural learning via Langevin dynamicsfor chaotic time series prediction', ICONIP 2017.
# (to be addeded on https://www.researchgate.net/profile/Rohitash_Chandra)

# ----------------------------------------------------------------------------------------------------------------------------------------
# Edited Gary Wong
# Decomposed MCMC Steps

# Randomize Populations
# For number of cycles on all sub-components,
#   Pick random indexes in population n
#       Create network individual by combining with best from other sub-populations (best defaulted to 0 index in all sub-populations)
#       Run MCMC for a number of samples
#       Get back individual, divide back into appropriate sections, replace current random index with refined weights
# Note: Elitism may be lost, traits from previous best solutions are lost due to randomized generation of proposal weights in MCMC



import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import os
import shutil






# An example of a class
class Network:
    def __init__(self, Topo, Train, Test):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        np.random.seed()

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def RMSE_Er(self, targets):
        return np.sqrt(((self.out - targets) ** 2).mean())

        # error = np.subtract(abs(self.out), abs(actualout))
        # # sqerror = np.sum(np.square(error)) / self.Top[2]
        # rootsqerror = np.sqrt(np.sum(np.square(error)) / (len(error) * self.Top[2]))
        # return rootsqerror

    # def RMSE_Er(self, actualout):
    #     error = np.subtract(abs(self.out), abs(actualout))
    #     #sqerror = np.sum(np.square(error)) / self.Top[2]
    #     rootsqerror = np.sqrt(np.sum(np.square(error)) / (len(error) * self.Top[2]))
    #     return rootsqerror

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        #sqerror = np.sum(np.square(error)) / len(error)
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired, vanilla):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]

    def evaluate_proposal(self, data, w):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.

        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for pat in xrange(0, size):
            Input[:] = data[pat, 0:self.Top[0]]
            Desired[:] = data[pat, self.Top[0]:]

            self.ForwardPass(Input)
            try:
                fx[pat] = self.out
            except:
               print 'Error'



        return fx


# --------------------------------------------------------------------------

# -------------------------------------------------------------------


class MCMC:
    def __init__(self, samples, traindata, testdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, data, w, tausq):
        y = data[:, self.topology[0]]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def sampler(self):

        # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]
        print y_train.size
        print y_test.size

        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)

        step_w = 0.02;  # defines how much variation you need in changes to w

        step_eta = 0.01;
        # --------------------- Declare FNN and initialize

        neuralnet = Network(self.topology, self.traindata, self.testdata)
        print 'evaluate Initial w'

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

        print likelihood

        naccept = 0
        print 'begin sampling using mcmc random walk'
        plt.plot(x_train, y_train)
        plt.plot(x_train, pred_train)
        plt.title("Plot of Data vs Initial Fx")
        plt.savefig('mcmcresults/begin.png')
        plt.clf()

        plt.plot(x_train, y_train)

        for i in range(samples - 1):

            w_proposal = w + np.random.normal(0, step_w, w_size)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood

            mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))

            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                print    i, ' is accepted sample'
                naccept += 1
                likelihood = likelihood_proposal
                prior_likelihood = prior_prop
                w = w_proposal
                eta = eta_pro

                print  likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted'

                pos_w[i + 1,] = w_proposal
                pos_tau[i + 1,] = tau_pro
                fxtrain_samples[i + 1,] = pred_train
                fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest

                plt.plot(x_train, pred_train)


                pred_train = neuralnet.evaluate_proposal(self.traindata, w)
                fitness = neuralnet.sampleEr(pred_train)
                print 'Fitness: ' + str(fitness)


            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]

                # print i, 'rejected and retained'

        print naccept, ' num accepted'
        print naccept / (samples * 1.0), '% was accepted'
        accept_ratio = naccept / (samples * 1.0) * 100

        plt.title("Plot of Accepted Proposals")
        plt.savefig('mcmcresults/proposals.png')
        plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
        plt.clf()

        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)



class Species:
    def __init__(self, PopulationSize, SpeciesSize):
        self.Populations = [SpeciesPopulation(SpeciesSize) for count in xrange(PopulationSize)]
        self.BestIndex = 0
        self.BestFitness = 0
        self.WorstIndex = 0
        self.WorstFitness = 0

        #Assign Initial Rank
        for i in xrange(len(self.Populations)):
            self.Populations[i].Rank = i

class SpeciesPopulation:
    def __init__(self, SpeciesSize):
        self.Likelihood = 0
        self.Fitness = 0
        self.Chromes = np.random.randn(SpeciesSize)
        self.Rank = 99
class CoEvolution:
    def __init__(self, Topology, PopulationSize, samples, traindata, testdata):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = Topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        self.populationsize = PopulationSize

        #SpeciesSize keeps track of the size of each species
        #  self.SpeciesSize = np.ones(Hidden + Output) # No species determined by no hidden and output neurons
        self.IndividualSize =  (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[1]) + self.topology[1] + self.topology[2]
        self.BestIndividual = np.zeros(self.IndividualSize)

        self.AllSpecies = []

        for i in xrange(self.topology[1]):
            x = Species(self.populationsize, self.topology[0] + 1)
            self.AllSpecies.append(x)
        for i in xrange(self.topology[2]):
            x = Species(self.populationsize, self.topology[1] + 1)
            self.AllSpecies.append(x)

    def PrintBestIndexes(self):
        tempStr = '['
        for i in range(len(self.AllSpecies)):
            tempStr += 'Species ' + str(i) + ': ' + str(self.AllSpecies[i].BestIndex) + ' '
        tempStr += ']'
        print tempStr

    def MarkBestInSubPopulations(self):
        for i in range(len(self.AllSpecies)):
            bestIndex = 0  # Set initial lowest
            bestFitness = 0
            for y in range(len(self.AllSpecies[i].Populations)):
                #if   self.AllSpecies[i].Populations[y].Fitness < bestFitness:
                if  self.AllSpecies[i].Populations[y].Fitness > bestFitness:
                    bestIndex = y
                    bestFitness = self.AllSpecies[i].Populations[y].Fitness

            self.AllSpecies[i].BestIndex = bestIndex
            self.AllSpecies[i].BestFitness = bestFitness

    def MarkWorstInSubPopulations(self):
        for i in range(len(self.AllSpecies)):
            worstIndex = 0  # Set initial lowest
            worstFitness = 0
            for y in range(len(self.AllSpecies[i].Populations)):
                #if   self.AllSpecies[i].Populations[y].Fitness < bestFitness:
                if  self.AllSpecies[i].Populations[y].Fitness > worstFitness:
                    worstIndex = y
                    worstFitness = self.AllSpecies[i].Populations[y].Fitness

            self.AllSpecies[i].WorstIndex = worstIndex
            self.AllSpecies[i].WorstFitness = worstFitness
    def OrderSubPopulations(self):
        #This function assumes that populations are newly generated with Consecutive ranks. Population[0] = Rank 0, Population[1] = Rank 1 and so on
        for i in range(len(self.AllSpecies)):
            #bestIndex = 0  # Set initial lowest
            #bestFitness = self.AllSpecies[i].Populations[0].Fitness  # Set initial lowest
            for y in range(len(self.AllSpecies[i].Populations) - 1):
                min_idx = y;
                for x in range(y + 1, len(self.AllSpecies[i].Populations)):
                    if self.AllSpecies[i].Populations[x].Fitness >self.AllSpecies[i].Populations[min_idx].Fitness:
                       min_idx = x
                if (self.AllSpecies[i].Populations[y].Rank > self.AllSpecies[i].Populations[min_idx].Rank ):  # Only swap ranks if next rank is less else leave as is
                    tempRank = self.AllSpecies[i].Populations[min_idx].Rank  # Take previous best index rank
                    self.AllSpecies[i].Populations[min_idx].Rank = self.AllSpecies[i].Populations[y].Rank  # Swap ranks
                    self.AllSpecies[i].Populations[y].Rank = tempRank



            #for x in range(len(self.AllSpecies[i].Populations)):
                    #  for y in range(len(self.AllSpecies[i].Populations) - 1 ):
                    #Check if next fitness is less than current. If it is, check rank and swap
                    #    if self.AllSpecies[i].Populations[y].Fitness < self.AllSpecies[i].Populations[y + 1].Fitness:
                        # Swap ranks
                    #     if(self.AllSpecies[i].Populations[y + 1].Rank < self.AllSpecies[i].Populations[y].Rank): #Only swap ranks if next rank is less else leave as is
                    #     tempRank = self.AllSpecies[i].Populations[y].Rank # Take previous best index rank
                    #    self.AllSpecies[i].Populations[y].Rank = self.AllSpecies[i].Populations[y + 1].Rank  # Swap ranks
                    #    self.AllSpecies[i].Populations[y + 1].Rank = tempRank
                    # else:
                        #Next fitness is less than current swap
                    #  if (self.AllSpecies[i].Populations[y + 1].Rank > self.AllSpecies[i].Populations[y].Rank):  # Only swap ranks if next rank is less else leave as is
                    #  tempRank = self.AllSpecies[i].Populations[y + 1].Rank  # Take previous best index rank
                    #  self.AllSpecies[i].Populations[y + 1].Rank = self.AllSpecies[i].Populations[ y].Rank  # Swap ranks
            #   self.AllSpecies[i].Populations[y].Rank = tempRank

            #Set best index and fitness
            for x in range(len(self.AllSpecies[i].Populations)):
                if self.AllSpecies[i].Populations[x].Rank == 0:
                    self.AllSpecies[i].BestIndex = x
                    self.AllSpecies[i].BestFitness = self.AllSpecies[i].Populations[x].Fitness


        print('Completed Population Sort by Fitness')
        #self.PrintPopulationFitness()
    def PrintPopulationFitness(self):

        #temp = ''
        #for i in range(len(self.AllSpecies)):  # Go through each species
          #  temp += 'Species ' + str(i) + ' | '

        temp2 = ''

        for y in range(self.populationsize):  #Go through each population
            temp = ''
            for i in range(len(self.AllSpecies)): #Go through each species
                if y == self.AllSpecies[i].BestIndex:
                    temp += 'F:' + str(self.AllSpecies[i].Populations[y].Fitness) + ' B |   '

                    for t in range(len(self.AllSpecies[i].Populations[y].Chromes)):
                        temp2 += str(self.AllSpecies[i].Populations[y].Chromes[t]) + ' '
                else:
                    temp += 'F:' + str(self.AllSpecies[i].Populations[y].Fitness) + '   |   '
            temp2 += '| '

            print temp
        print temp2

    def EvaluateSpeciesByFitness(self, neuralnet):

        # Go through all SpeciesPopulation
        #   Conjoin with best from other Species
        #   Perform a forward pass and assign fitness
        individual = []

        for i in range(len(self.AllSpecies)):
            for popindex in range(len(self.AllSpecies[i].Populations)):
                [individual, speciesRange] = self.GetIndividualCombinedWithBest(i,popindex)
                pred_train = neuralnet.evaluate_proposal(self.traindata, self.ToMCMCIndividual(individual))
                #fitness = neuralnet.sampleEr(pred_train)
                fitness = neuralnet.RMSE_Er(pred_train)
                self.AllSpecies[i].Populations[popindex].Fitness = fitness
                individual = [] # reset individual
            print 'Evaluated Species: ' + str(i)

        self.MarkBestInSubPopulations()
        #self.PrintPopulationFitness()
        #self.OrderSubPopulations()

       # w = self.GetBestIndividual()
       # pred_train = neuralnet.evaluate_proposal(self.traindata, w)
       # fitness = neuralnet.sampleEr(pred_train)

    def GetIndividualCombinedWithBest(self, SpeciesIndex, PopulationIndex):
        combinedWithBestIndividual = []
        speciesRange = np.zeros((2,), dtype=np.int)  # start and end index in individual
        speciesFound = 0
        index = 0
        for u in range(len(self.AllSpecies)):
            if u == SpeciesIndex: #Only replace part of the best individual where this species is

                if speciesFound == 0: #If this is the first chrome for the current species, mark starting index
                    speciesRange[0] = index #Set beginning index
                    speciesFound = 1

                for chrome in (self.AllSpecies[u].Populations[PopulationIndex].Chromes):
                    combinedWithBestIndividual.append(chrome)
                    index += 1

            else:
                for chrome in (self.AllSpecies[u].Populations[self.AllSpecies[u].BestIndex].Chromes):
                    combinedWithBestIndividual.append(chrome) #Append best from other species
                    index += 1

        speciesRange[1] = speciesRange[0] + (len(self.AllSpecies[SpeciesIndex].Populations[0].Chromes) - 1) # Set endpoint

        return [combinedWithBestIndividual, speciesRange]
    def ExtractPopulationFromIndividual(self, SpeciesIndex, Individual):
        population = []
        index = 0
        for u in range(len(self.AllSpecies)):
            if u == SpeciesIndex:
                for chrome in (self.AllSpecies[u].Populations[0].Chromes):
                    population.append(Individual[index])
                    index += 1
                break

            else:
                index += len(self.AllSpecies[u].Populations[0].Chromes)
        return population

    def ReplaceRandomIndexesInSpecies(self, Individual):
        index = 0
        for u in range(len(self.AllSpecies)):
            randomIndex = random.randint(0, self.populationsize - 1)  # Get random index in population
            for chrome in (self.AllSpecies[u].Populations[randomIndex].Chromes):
                chrome = Individual[index]
                index += 1
            self.AllSpecies[u].BestIndex = randomIndex

    def ReplaceWorstIndexesInSpecies(self, Individual):

        self.MarkWorstInSubPopulations()

        index = 0
        for u in range(len(self.AllSpecies)):

            worstIndex = self.AllSpecies[u].WorstIndex
            for chrome in (self.AllSpecies[u].Populations[worstIndex].Chromes):
                chrome = Individual[index]
                index += 1

    def GetBestIndividual(self):
        # Make first individual the best
        bestIndividual = []
        for i in range(len(self.AllSpecies)):
            for y in range(len(self.AllSpecies[i].Populations[self.AllSpecies[i].BestIndex].Chromes)):
                bestIndividual.append(self.AllSpecies[i].Populations[self.AllSpecies[i].BestIndex].Chromes[y])
        self.BestIndividual = bestIndividual
        return bestIndividual

    # ------------------------------------------------------------------------------------------------------------------
    #   This method updates n number of populations in a species
    #   It combines each population with the best from other populations
    #   Performs MCMC for a number of samples
    #   Breaks the new weights back into species size
    #   Replaces currently selected population with updated weights from MCMC
    #   Evaluate fitness, assign fitness and recheck best fitness in population
    # ------------------------------------------------------------------------------------------------------------------

    def PopulationsRandomWalk_MCMC_Populations(self, NumIndexes, Cycles, NeuralNet):
        dmcmc = DecomposedMCMC(self.samples, self.traindata, self.testdata, self.topology)

        weights = []
        bestWeightsOutputInCycles_Train = []
        bestWeightsOutputInCycles_Test = []
        weightsAtEndOfCycles = []
        fitnessAtEndOfCycles = []
        populationDataAtEndOfCycles = []

        for h in range(Cycles):
            print 'Cycle ' + str(h)
            for u in range(len(self.AllSpecies)):
                #print 'Species ' + str(u) + ' ----------------------------------'
                for i in range(NumIndexes):
                    #print 'Index ' + str(i)

                    randomIndex = random.randint(0, self.populationsize - 1) #Get random index in population
                    #1. Update random index chromes
                    #2. Get fitness - combine with unchanged best

                    [individual, speciesRange] = self.GetIndividualCombinedWithBest(u,randomIndex)
                    modifiedIndividual = dmcmc.sampler(self.ToMCMCIndividual(individual), NeuralNet, 0)
                    #modifiedIndividual = dmcmc.samplerPopulationProposals(individual, NeuralNet, 0, speciesRange)

                    # ytraindata = self.traindata[:, self.topology[0]]
                    extractedChromes = self.ExtractPopulationFromIndividual(u, self.ToDecomposedIndividual(modifiedIndividual))
                    # Replace random index population
                    self.AllSpecies[u].Populations[randomIndex].Chromes = extractedChromes
                    # Update fitness by performing a forward pass
                    [ind, speciesRange] = self.GetIndividualCombinedWithBest(u, randomIndex)
                    pred_train = NeuralNet.evaluate_proposal(self.traindata, self.ToMCMCIndividual(ind))
                    #fitness = NeuralNet.sampleEr(pred_train)
                    fitness = NeuralNet.RMSE_Er(pred_train)
                    self.AllSpecies[u].Populations[randomIndex].Fitness = fitness



                #After every species, evaluate best in species
                tempfit = self.AllSpecies[u].Populations[0].Fitness

                for t in range(1, self.populationsize):
                    if(self.AllSpecies[u].Populations[t].Fitness > tempfit):
                        tempfit = self.AllSpecies[u].Populations[t].Fitness
                        self.AllSpecies[u].BestIndex = t
                        self.AllSpecies[u].BestFitness = tempfit


                    #Assign fitness?

            #Evaluate all species
            #self.EvaluateSpeciesByFitness(NeuralNet) #Assign new fitness
            #self.MarkBestInSubPopulations() #Mark fittest populations

            #Show decreasing fitness
            individual = self.GetBestIndividual()
            pred_train = NeuralNet.evaluate_proposal(self.traindata, self.ToMCMCIndividual(individual))
            pred_test = NeuralNet.evaluate_proposal(self.testdata, self.ToMCMCIndividual(individual))
            #fitness = NeuralNet.sampleEr(pred_train)
            fitness = NeuralNet.RMSE_Er(pred_train)
            print 'Fitness: ' + str(fitness)
            self.PrintBestIndexes()

            #These are for graph data at the end of the simulation
            weightsAtEndOfCycles.append(self.ToMCMCIndividual(individual))
            fitnessAtEndOfCycles.append(fitness)
            populationDataAtEndOfCycles.append(self.AllSpecies)
            bestWeightsOutputInCycles_Train.append(pred_train)
            bestWeightsOutputInCycles_Test.append(pred_test)

        weights.append(bestWeightsOutputInCycles_Train) #Output[0]
        weights.append(bestWeightsOutputInCycles_Test)  #Output[1]
        weights.append(weightsAtEndOfCycles)            #Output[2]
        weights.append(fitnessAtEndOfCycles)            #Output[3]
        weights.append(populationDataAtEndOfCycles)     #Output[4]

        return weights




    # ------------------------------------------------------------------------------------------------------------------
    #   This method randomizes weights in all populations in initialization
    #   Initial fitness evaluation is done and fitness assigned. BestIndexes and BestFitness are also set
    #   in all subpopulations]
    #   i. Take out best and combine into best individual
    #   Run MCMC for n samples
    #   Divide back up
    #   Replace random index and assign as best indexes
    #   Foreach population in all species, reevaluate fitness by combining with new best
    #   Update fittest in all subpopulations
    #   Restart i.

    # ------------------------------------------------------------------------------------------------------------------

    def PopulationsRandomWalk_MCMC_BestOnly(self, NumIndexes, Cycles, NeuralNet):
        dmcmc = DecomposedMCMC(self.samples, self.traindata, self.testdata, self.topology)

        for h in range(Cycles):
            print 'Cycle' + str(h)

            bestInd = self.GetBestIndividual()
            modifiedIndividual = dmcmc.sampler(bestInd, NeuralNet,0)
            #Break up and replace random indexes in sub-populations
            #self.ReplaceRandomIndexesInSpecies(modifiedIndividual)
            self.ReplaceWorstIndexesInSpecies(modifiedIndividual)
            self.EvaluateSpeciesByFitness(NeuralNet) #Assign new fitness
            self.MarkBestInSubPopulations() #Mark fittest populations

            #Show decreasing fitness
            individual = self.GetBestIndividual()
            pred_train = NeuralNet.evaluate_proposal(self.traindata, individual)
            #fitness = NeuralNet.sampleEr(pred_train)
            fitness = NeuralNet.RMSE_Er(pred_train)
            print 'Fitness: ' + str(fitness)
            self.PrintBestIndexes()

    def ToDecomposedIndividual(self, ind):

        individual = None
        if type(ind) is np.ndarray:
            individual = ind.tolist()
        else:
            individual = ind

        WeightsSize = (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[2])
        BiasSize = self.topology[1] + self.topology[2]
        Decomposed_Individual = []
        current_index = 0

        Species_Weights = []
        Weights = individual[0:WeightsSize]
        Biases = individual[WeightsSize:WeightsSize + BiasSize]

        for i in range(len(self.AllSpecies)):
            fromIndex = current_index
            endIndex = fromIndex + len(self.AllSpecies[i].Populations[0].Chromes) - 1
            Species_Weights.append(individual[fromIndex:endIndex])
            current_index = endIndex

        for i in range(len(Biases)):
            Species_Weights[i].append(Biases[i])

        # Create final individual
        for i in range(len(Species_Weights)):
            for t in range(len(Species_Weights[i])):
                Decomposed_Individual.append(Species_Weights[i][t])

        return Decomposed_Individual

    def ToMCMCIndividual(self, individual):
        Weights = []
        Biases = []
        current_index = 0
        MCMC_Individual = []

        for i in range(len(self.AllSpecies)):
            fromIndex = current_index
            endIndex = fromIndex + len(self.AllSpecies[i].Populations[0].Chromes) - 1
            Weights.append(individual[fromIndex:endIndex])
            Biases.append(individual[endIndex])
            current_index = endIndex + 1

        for i in range(len(Weights)):
            for t in range(len(Weights[i])):
                MCMC_Individual.append(Weights[i][t])

        for i in range(len(Biases)):
            MCMC_Individual.append(Biases[i])

        return MCMC_Individual

class DecomposeProcedure:
    def __init__(self, samples, traindata, testdata, topology, populationSize):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        self.populationSize = populationSize

    #-----------------------------------------------------------#
    #            Main Decomposed MCMC Procedure                 #
    # ----------------------------------------------------------#

    def Procedure(self, run):
        neuronLevel = CoEvolution(self.topology, self.populationSize, self.samples, self.traindata, self.testdata )
        #neuronLevel.EvaluateSpecies()



        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]

        neuralnet = Network(self.topology, self.traindata, self.testdata)

        #   Initially, best index are all set to 0. During initial fitness evaluation lets run MCMC for a number of samples
        #   where we combine all index 0 into an individual and run to initialize direction
        dmcmc = DecomposedMCMC(self.samples, self.traindata, self.testdata, self.topology)
        ini = neuronLevel.GetBestIndividual()

        # temp1 = neuronLevel.ToMCMCIndividual(ini)
        # temp2 = neuronLevel.ToDecomposedIndividual(temp1)


        # initial_w = dmcmc.sampler(ini, neuralnet, 1000) #Update initial best
        neuronLevel.EvaluateSpeciesByFitness(neuralnet)



        output = neuronLevel.PopulationsRandomWalk_MCMC_Populations(20,20,neuralnet) #(NumIndexes, Cycles, NeuralNet):
        #neuronLevel.PopulationsRandomWalk_MCMC_BestOnly(10, 20, neuralnet)  # (NumIndexes, Cycles, NeuralNet):


        self.PlotGraphs(neuronLevel.ToMCMCIndividual(neuronLevel.GetBestIndividual()), neuralnet,neuronLevel.ToMCMCIndividual(ini),run, output)

        neuronLevel.PrintPopulationFitness()
        # self.PlotGraphs(initial_w, neuralnet, ini)

        return output[3][len(output[3]) - 1] # return final population best fitness

    def PlotGraphs(self, w, net, initial_w, run, output):
        print 'New and Old Weights '
        print w
        print initial_w
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        ytestdata = self.testdata[:, self.topology[0]]
        ytraindata = self.traindata[:, self.topology[0]]

        initial_trainout = net.evaluate_proposal(self.traindata,initial_w)
        trainout = net.evaluate_proposal(self.traindata,w)

        # print 'Initial Trainout'
        # print initial_trainout
        # print 'Trainout'
        # print trainout


        initial_testout = net.evaluate_proposal(self.testdata, initial_w)
        testout = net.evaluate_proposal(self.testdata,w)

        #Print train

        for i in range(len(output[0])):
            plt.plot(x_train, output[0][i], label='', color="black", lw=0.5, ls = '-')

        plt.plot(x_train, ytraindata, label='actual')
        plt.plot(x_train, trainout, label='predicted (train)')
        plt.plot(x_train, initial_trainout, label='initial (train)')




        plt.legend(loc='upper right')
        plt.title("Plot of Train Data vs MCMC Uncertainty ")

        if not os.path.exists('mcmcresults/run' + str(run) + '/'):
            os.makedirs('mcmcresults/run' + str(run)+ '/')

        plt.savefig('mcmcresults/run' + str(run) +'/dmcmc_train.png')
        plt.savefig('mcmcresults/run' + str(run) +'/dmcmc_train.svg', format='svg', dpi=600)
        plt.clf()

        # Print test

        for i in range(len(output[1])):
            plt.plot(x_test, output[1][i], label='', color="black", lw=0.5, ls = '-')

        plt.plot(x_test, ytestdata, label='actual')
        plt.plot(x_test, testout, label='predicted (test)')
        plt.plot(x_test, initial_testout, label='initial (test)')
        plt.legend(loc='upper right')
        plt.title("Plot of Test Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/run' + str(run) +'/dmcmc_test.png')
        plt.savefig('mcmcresults/run' + str(run) +'/dmcmc_test.svg', format='svg', dpi=600)
        plt.clf()


        colors = (0, 0, 0)
        area = np.pi * 3

        # Plot Weights


        for i in range(len(output[2])):

            # Plot Best Individual at the end of cycle
            x_weights = np.linspace(0, 1, num=len(output[2][0]))
            plt.plot(x_weights, output[2][i],'bo', x_weights, output[2][i],'k',label='')
            plt.legend(loc='upper right')
            plt.title('Best Solution at the end of [Cycle: ' + str(i) + ' with Fitness: ' + format(output[3][i], '.5f') + ']')
            plt.savefig('mcmcresults/run' + str(run) + '/weightsCycle'+str(i)+'.png')
            plt.savefig('mcmcresults/run' + str(run) + '/weightsCycle'+str(i)+'.svg', format='svg', dpi=600)
            plt.clf()

            # Plot population data at the end of cycle

            #----------------------------------------------------------------
            # outputCyclePopulations = 1
            #
            # if outputCyclePopulations == 1:
            #     clust_data = []
            #
            #     for y in range(self.populationSize):  # Go through each population
            #         temp = []
            #         for l in range(len(output[4][i])):  # Go through each species
            #             if y == output[4][i][l].BestIndex:
            #                 temp.append('F:' + str(output[4][i][l].Populations[y].Fitness) + ' B ')
            #             else:
            #                 temp.append('F:' + str(output[4][i][l].Populations[y].Fitness))
            #
            #         clust_data.append(temp)
            #
            #     colLabels = []
            #     for p in range(len(output[4][i])):
            #         colLabels.append('Species ' + str(p))
            #
            #     nrows, ncols = len(clust_data) + 1, len(colLabels)
            #     hcell, wcell = 0.3, 1.
            #     hpad, wpad = 0, 0
            #     fig = plt.figure(figsize=(ncols * wcell + wpad, nrows * hcell + hpad))
            #     ax = fig.add_subplot(111)
            #     ax.axis('off')
            #     # do the table
            #     the_table = ax.table(cellText=clust_data,
            #                          colLabels=colLabels,
            #                          loc='center')
            #     plt.savefig('mcmcresults/run' + str(run) + '/populationsData' + str(i) + '.png')
            #     plt.savefig('mcmcresults/run' + str(run) + '/populationsData' + str(i) + '.svg', format='svg', dpi=600)
            #     plt.close(fig)
            #     plt.clf()

            # ----------------------------------------------------------------



class DecomposedMCMC:
    def __init__(self, samples, traindata, testdata, topology):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, data, w, tausq):
        y = data[:, self.topology[0]]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def sampler(self, Individual, neuralnet, initialsamples):
        #print 'Begin sampling '
        # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]
        #print y_train.size
        #print y_test.size

        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        #w = np.random.randn(w_size)
        w = Individual
        w_proposal = np.random.randn(w_size)

        #step_w = 0.02;  # defines how much variation you need in changes to w
        step_w = 0.1;  # defines how much variation you need in changes to w
        step_eta = 0.01;
        # --------------------- Declare FNN and initialize


        # print 'evaluate Initial w'

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w,
                                                 tau_pro)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

        #print likelihood

        #naccept = 0
        #print 'begin sampling using mcmc random walk'
        #plt.plot(x_train, y_train)
        #plt.plot(x_train, pred_train)
        #plt.title("Plot of Data vs Initial Fx")
        #plt.savefig('mcmcresults/begin.png')
        # plt.clf()

        # plt.plot(x_train, y_train)

        samp = samples

        if initialsamples != 0:
            samp = initialsamples

        for i in range(samp - 1):
            w_proposal = w + np.random.normal(0, step_w, w_size)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata,
                                                                                w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata,
                                                                            w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood

            try:
                mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))

                u = random.uniform(0, 1)

                if u < mh_prob:
                    # Update position
                    #print    i, ' is accepted sample'
                    #naccept += 1
                    likelihood = likelihood_proposal
                    prior_likelihood = prior_prop
                    w = w_proposal
                    eta = eta_pro


                    # pred_train = neuralnet.evaluate_proposal(self.traindata, w)
                    # # fitness = NeuralNet.sampleEr(pred_train)
                    # fitness = neuralnet.RMSE_Er(pred_train)
                    # print 'Fitness: ' + str(fitness)

                    # pred_train = neuralnet.evaluate_proposal(self.traindata, w)
                    # fitness = neuralnet.RMSE_Er(pred_train)
                    print 'Fitness: ' + str(rmsetrain)

                    # fxtrain_samples[i + 1,] = pred_train
                    # fxtest_samples[i + 1,] = pred_test
            except Exception:
                print '######################### OverflowError: math range error: Skipping current sample'
                pass  # or you could use 'continue'


        return w  # return as soon as we get an accepted sample

    def samplerPopulationProposals(self, Individual, neuralnet, initialsamples, speciesRange):

        # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]
        #print y_train.size
        #print y_test.size

        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        #w = np.random.randn(w_size)
        w = Individual
        w_proposal = self.replaceChromesIndividual(w, np.random.randn(w_size), speciesRange)

        #step_w = 0.02;  # defines how much variation you need in changes to w
        step_w = 0.002;  # defines how much variation you need in changes to w
        step_eta = 0.01;
        # --------------------- Declare FNN and initialize


        # print 'evaluate Initial w'

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w,
                                                 tau_pro)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

        #print likelihood

        #naccept = 0
        #print 'begin sampling using mcmc random walk'
        #plt.plot(x_train, y_train)
        #plt.plot(x_train, pred_train)
        #plt.title("Plot of Data vs Initial Fx")
        #plt.savefig('mcmcresults/begin.png')
        # plt.clf()

        # plt.plot(x_train, y_train)

        samp = samples

        if initialsamples != 0:
            samp = initialsamples

        for i in range(samp - 1):
            #w_proposal = w +
            w_proposal = self.replaceChromesIndividual(w, np.random.normal(0, step_w, w_size), speciesRange) #(old,new,range)


            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata,
                                                                                w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata,
                                                                            w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood
            # print 'diff_likelihood: ' + str(diff_likelihood)
            # print 'diff_priorliklihood: ' + str(diff_priorliklihood)

            try:
                mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))

                u = random.uniform(0, 1)

                if u < mh_prob:
                    # Update position
                    # print    i, ' is accepted sample'
                    # naccept += 1
                    likelihood = likelihood_proposal
                    prior_likelihood = prior_prop
                    w = w_proposal
                    eta = eta_pro

                    # pred_train = neuralnet.evaluate_proposal(self.traindata, w)
                    # # fitness = NeuralNet.sampleEr(pred_train)
                    # fitness = neuralnet.RMSE_Er(pred_train)
                    # print 'Fitness: ' + str(fitness)

                    # pred_train = neuralnet.evaluate_proposal(self.traindata, w)
                    # fitness = neuralnet.sampleEr(pred_train)
                    # print 'Fitness: ' + str(fitness)
            except Exception:
                print '######################### OverflowError: math range error: Skipping current sample'
                pass  # or you could use 'continue'

        return w  # return as soon as we get an accepted sample

    #This function takes in two individuals, a previous and a new. It replaces all chromes from starting index (speciesRange[0]) to end index (speciesRange[1])
    def replaceChromesIndividual(self, originalIndividual, newIndividual, speciesRange):
        for i in range(speciesRange[0], speciesRange[1]):
            originalIndividual[i] = newIndividual[i]

        return originalIndividual


def main():
    if os.path.exists('mcmcresults'):
        shutil.rmtree('mcmcresults/', ignore_errors=True)
        os.makedirs('mcmcresults')

    else:
        os.makedirs('mcmcresults')

        # shutil.rmtree('mcmcresults/', ignore_errors=True)

    # os.makedirs('mcmcresults')

    start = time.time()


    outres = open('mcmcresults/resultspriors.txt', 'w')
    hidden = 5
    input = 4  #
    output = 1

    populationSize = 50
    traindata = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
    testdata = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt")  #
    # if problem == 1:
    #     traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
    #     testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")  #
    # if problem == 2:

    # if problem == 3:
    #     traindata = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")
    #     testdata = np.loadtxt("Data_OneStepAhead/Mackey/test.txt")  #

    print(traindata)

    topology = [input, hidden, output]

    MinCriteria = 0.005  # stop when RMSE reaches MinCriteria ( problem dependent)

    random.seed(time.time())

    numSamples = 20  # need to decide yourself 80000

    # 2 for population
    #1000 for bestonly
    # for best only increase poulation size

    runs = 10

    RUNRMSE = []

    for i in range(0, runs):
        strategy = DecomposeProcedure(numSamples, traindata, testdata, topology, populationSize)
        RUNRMSE.append(strategy.Procedure(i))

    Errors = np.zeros(runs).tolist()
    # Errors = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    x_train = range(0, (runs + 1))
    y_pos = np.arange((runs + 1))

    MeanRMSE = np.mean(RUNRMSE)
    MeanRMSESTD = np.std(RUNRMSE)

    RUNRMSE.append(MeanRMSE)
    Errors.append(MeanRMSESTD)

    plt.bar(y_pos, RUNRMSE, align='center', alpha=0.5)
    plt.errorbar(
        x_train,  # X
        RUNRMSE,  # Y
        yerr=Errors,  # Y-errors
        label="Error bars plot",
        fmt="gs",  # format line like for plot()
        linewidth=1  # width of plot line
    )
    plt.xticks(y_pos, x_train)
    plt.ylabel('Train RMSE')
    plt.ylabel('Run')
    plt.title('Programming language usage')
    plt.savefig('mcmcresults/AverageResults.png')
    plt.savefig('mcmcresults/AverageResults.svg', format='svg', dpi=600)
    plt.clf()

    # Plot Average RUN RMSE


    # mcmc = MCMC(numSamples, traindata, testdata, topology)  # declare class
    #
    # [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler()
    # print 'sucessfully sampled'
    #
    # burnin = 0.1 * numSamples  # use post burn in samples
    #
    # pos_w = pos_w[int(burnin):, ]
    # pos_tau = pos_tau[int(burnin):, ]
    #
    # fx_mu = fx_test.mean(axis=0)
    # fx_high = np.percentile(fx_test, 95, axis=0)
    # fx_low = np.percentile(fx_test, 5, axis=0)
    #
    # fx_mu_tr = fx_train.mean(axis=0)
    # fx_high_tr = np.percentile(fx_train, 95, axis=0)
    # fx_low_tr = np.percentile(fx_train, 5, axis=0)
    #
    # rmse_tr = np.mean(rmse_train[int(burnin):])
    # rmsetr_std = np.std(rmse_train[int(burnin):])
    # rmse_tes = np.mean(rmse_test[int(burnin):])
    # rmsetest_std = np.std(rmse_test[int(burnin):])
    # print rmse_tr, rmsetr_std, rmse_tes, rmsetest_std
    # np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')
    #
    # ytestdata = testdata[:, input]
    # ytraindata = traindata[:, input]
    #
    # plt.plot(x_test, ytestdata, label='actual')
    # plt.plot(x_test, fx_mu, label='pred. (mean)')
    # plt.plot(x_test, fx_low, label='pred.(5th percen.)')
    # plt.plot(x_test, fx_high, label='pred.(95th percen.)')
    # plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
    # plt.legend(loc='upper right')
    #
    # plt.title("Plot of Test Data vs MCMC Uncertainty ")
    # plt.savefig('mcmcresults/mcmcrestest.png')
    # plt.savefig('mcmcresults/mcmcrestest.svg', format='svg', dpi=600)
    # plt.clf()
    # # -----------------------------------------
    # plt.plot(x_train, ytraindata, label='actual')
    # plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
    # plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
    # plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
    # plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
    # plt.legend(loc='upper right')
    #
    # plt.title("Plot of Train Data vs MCMC Uncertainty ")
    # plt.savefig('mcmcresults/mcmcrestrain.png')
    # plt.savefig('mcmcresults/mcmcrestrain.svg', format='svg', dpi=600)
    # plt.clf()
    #
    # mpl_fig = plt.figure()
    # ax = mpl_fig.add_subplot(111)
    #
    # ax.boxplot(pos_w)
    #
    # ax.set_xlabel('[W1] [B1] [W2] [B2]')
    # ax.set_ylabel('Posterior')
    #
    # plt.legend(loc='upper right')
    #
    # plt.title("Boxplot of Posterior W (weights and biases)")
    # plt.savefig('mcmcresults/w_pos.png')
    # plt.savefig('mcmcresults/w_pos.svg', format='svg', dpi=600)
    #
    # plt.clf()

    print 'End simulation'
    end = time.time()
    print str(end - start) + ' Seconds'


if __name__ == "__main__": main()
