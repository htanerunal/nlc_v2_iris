//
//  main.cpp
//  Neural Logic Circuits - NLC - v2 - Multi-Class Classification
//
//  Created by Hamit Taner Ünal on 19.03.2023.
//  Copyright © 2021 Dr.Hamit Taner Ünal & Prof.Fatih Başçiftçi. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <array>
#include <sys/time.h>
#include <stdlib.h>
#include <cmath>

using namespace std;

inline const char * const BoolToString(bool b);
void write_csv(std::string filename, std::string colname, std::vector<int> vals);
int random(int from, int to);
void swap (bool *a, bool *b);
void randomShuffle (bool arr[], int n);
void swapInt (int *a, int *b);
void randomShuffleInt (int arr[], int n);
void printArray (bool arr[], int n, int numberofinputbits);
bool getAndMask(bool maskArray[], bool outputArray[], int n);
bool getOrMask(bool maskArray[], bool outputArray[], int n);
bool getNAndMask(bool maskArray[], bool outputArray[], int n);
bool getNOrMask(bool maskArray[], bool outputArray[], int n);
bool getXOrMask(bool maskArray[], bool outputArray[], int n);
bool getXNOrMask(bool maskArray[], bool outputArray[], int n);
int findMax(float arr[], int n, int gateCount[], int popIndex[]);
int findMaxSourceIndex(int inputArray[], int arrayLength);
int findMaxSourceValue(int inputArray[], int arrayLength);
int roundP(float d);
int powerF(int x, int y);
int getClass(bool arr[], int n);
int seed;


int main(int argc, const char * argv[]) {

    //Randomize device (get best random numbers)
    struct timeval time;
    gettimeofday(&time,NULL);

    //Randomize with a seed
    seed = 1;
    srand(seed);

    //Dataset dependent variables
    int outputCount=2;//Bit length of output class
    int numberOfClasses = 4;//Max value for output bits
    bool reportTrainNetwork = false;//Report interim results of training (could make the process slower)


    //Read CSV File and collect info on data
    std::ifstream myFile("dataset.csv");
    std::string line, colName;
    int val;
    int colNumber=0;
    int numberOfTotalRows = 0;

    //Get column names, number of columns, number of rows
    if(myFile.good())
    {
        // Extract the first line in the file
        std::getline(myFile, line);

        // Create a stringstream from line
        std::stringstream ss(line);

        printf("Column Names\n");
        // Extract each column name
        while(std::getline(ss, colName, ',')){
            colNumber++;

            // Print colnames
            printf("%s", colName.c_str());
            printf(", ");
        }
        printf("\n");

        //Get row count:
        std::string unused;
        while ( std::getline(myFile, unused) )
            ++numberOfTotalRows;
    }
    myFile.close();
    int numberOfInputBits = colNumber - outputCount;//except column 0

    //Now, let's print this info
    printf("Number of input bits:");printf("%d", numberOfInputBits);printf("\n");
    printf("Number of rows:");printf("%d", numberOfTotalRows);printf("\n");

    //Define raw data variables
    bool X_raw[numberOfTotalRows][numberOfInputBits];
    bool y_raw[numberOfTotalRows][outputCount];
    int y_class[numberOfTotalRows];//Encoded integer for any categorical variable

    //Read the file again and fill dataset variables with raw data
    std::ifstream myFileAgain("dataset.csv");
    std::getline(myFileAgain, line);

    int rowID=0;
    // Read data, line by line
    while(std::getline(myFileAgain, line))
    {
        // Create a string stream of the current line
        std::stringstream ss(line);
        // Keep track of the current column index
        int colIdx = 0;
        // Extract each integer
        while(ss >> val){
            if (colIdx < numberOfInputBits) {
                X_raw[rowID][colIdx] = (bool)val;
            }
            else {
                y_raw[rowID][colIdx-numberOfInputBits] = (bool)val;
            }
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            // Increment the column index
            colIdx++;
        }
        rowID++;
    }
    // Close file
    myFileAgain.close();

    //Read classes of categorical outputs
    for (int i=0;i<numberOfTotalRows;i++)
    {
        y_class[i]=0;
        for (int j=0;j<outputCount;j++)
        {
            y_class[i]+=(int)y_raw[i][j]*(powerF(2,(outputCount-j-1)));
        }
    }

    //Define evaluation method (kFold or train-test split)
    bool isKFold = false;

    //kFold parameters
    int numberOfK = 5;
    int currentFold=0;
    //**** Important Note *******
    //kFold cross validation is managed MANUALLY
    //You have to change the current fold on each run.
    //Record results and calculate MEAN/STD manually

    //Train-Test Split variables
    float train_test_split = 0.8;

    //Generic variables for train-test chunks
    int numberOfTrainSamples;
    int numberOfTestSamples;

    //Bit-level variables to store data
    bool X_train[numberOfTotalRows][numberOfInputBits];
    bool y_train[numberOfTotalRows][outputCount];

    bool X_test[numberOfTotalRows][numberOfInputBits];
    bool y_test[numberOfTotalRows][outputCount];

    int y_train_class[numberOfTotalRows];
    int y_test_class[numberOfTotalRows];

    //Start splitting the data
    //Split data for cross-validation
    if (isKFold)
    {
        //Determine standard test size
        int testStandardSampleCount = roundP((float)numberOfTotalRows/(float)numberOfK);
        //For the last K-fold, set the size as the remaining samples
        if (currentFold<numberOfK-1) numberOfTestSamples = testStandardSampleCount;
        else numberOfTestSamples = (numberOfTotalRows - ((numberOfK-1)*testStandardSampleCount));
        //Determine train size
        numberOfTrainSamples = numberOfTotalRows - numberOfTestSamples;
        //Print summary on the current fold
        printf("K-Fold Data----------------------\n");
        printf("Current K fold:%d\n",currentFold);
        printf("Number of train samples:");printf("%d",numberOfTrainSamples);printf("\n");
        printf("Number of test samples:");printf("%d",numberOfTestSamples);printf("\n");

        //Shuffle raw data by creating a shuffle index
        int indexArray[numberOfTotalRows];
        for (int i=0;i<numberOfTotalRows;i++) indexArray[i] = i;
        randomShuffleInt(indexArray,numberOfTotalRows);

        //fill train rows (X)
        int currentTrainRow = 0;
        for (int x=0;x<(currentFold*testStandardSampleCount);x++)
        {
            for (int c=0;c<numberOfInputBits;c++)
            {
                X_train[currentTrainRow][c] = X_raw[indexArray[x]][c];
            }
            currentTrainRow++;
        }

        //Skip test fold and continue filling train rows (X)
        if (currentFold!=numberOfK-1) {
            for (int x = ((currentFold + 1) * testStandardSampleCount); x < numberOfTotalRows; x++) {
                for (int c = 0; c < numberOfInputBits; c++) {
                    X_train[currentTrainRow][c] = X_raw[indexArray[x]][c];
                }
                currentTrainRow++;
            }
        }

        //Fill train rows (y)
        currentTrainRow = 0;
        for (int y=0;y<(currentFold*testStandardSampleCount);y++)
        {
            for (int c=0;c<outputCount;c++)
            {
                y_train[currentTrainRow][c] = y_raw[indexArray[y]][c];
            }
            y_train_class[currentTrainRow] = y_class[indexArray[y]];
            currentTrainRow++;
        }

        //Skip test fold and continue filling train rows (y)
        if (currentFold!=numberOfK-1) {
            for (int y = ((currentFold + 1) * testStandardSampleCount); y < numberOfTotalRows; y++) {
                for (int c = 0; c < outputCount; c++) {
                    y_train[currentTrainRow][c] = y_raw[indexArray[y]][c];
                }
                y_train_class[currentTrainRow] = y_class[indexArray[y]];
                currentTrainRow++;
            }
        }


        //fill test rows
        int currentTestRow=0;
        for (int x=(currentFold*testStandardSampleCount);x<((currentFold*testStandardSampleCount) + numberOfTestSamples);x++)
        {
            for (int c=0;c<numberOfInputBits;c++)
            {
                X_test[currentTestRow][c] = X_raw[indexArray[x]][c];
            }
            for (int c=0;c<outputCount;c++)
            {
                y_test[currentTestRow][c] = y_raw[indexArray[x]][c];
            }
            y_test_class[currentTestRow] = y_class[indexArray[x]];
            currentTestRow++;
        }

        //Print shuffled dataset (FULL)
        printf("\n");
        printf("Printing Shuffled Dataset\n");
        for (int i=0;i<numberOfTotalRows;i++)
        {
            printf("INDEX:%d DATA x: ",indexArray[i]);
            for (int c=0;c<numberOfInputBits;c++)
            {
                printf("%d ",X_raw[indexArray[i]][c]);
            }
            printf(" y: ");
            for (int c=0;c<outputCount;c++)
            {
                printf("%d ",y_raw[indexArray[i]][c]);
            }
            printf(" class: ");
            printf("%d",y_class[indexArray[i]]);
            printf("\n");
        }
        //Now, print train data
        printf("\n");
        printf("TRAIN:\n");
        currentTrainRow = 0;
        for (int i=0;i<(currentFold*testStandardSampleCount);i++)
        {
            printf("INDEX:%d DATA x: ",indexArray[i]);
            for (int c=0;c<numberOfInputBits;c++)
            {
                printf("%d ",X_train[currentTrainRow][c]);
            }
            printf(" y: ");
            for (int c=0;c<outputCount;c++)
            {
                printf("%d ",y_train[currentTrainRow][c]);
            }
            printf(" class: ");
            printf("%d",y_train_class[currentTrainRow]);
            printf("\n");
            currentTrainRow++;
        }
        //Skip test fold and continue printing train data
        if (currentFold!=numberOfK-1)
        {
            for (int i = ((currentFold + 1) * testStandardSampleCount); i < numberOfTotalRows; i++)
            {
                printf("INDEX:%d DATA x: ",indexArray[i]);
                for (int c=0;c<numberOfInputBits;c++)
                {
                    printf("%d ",X_train[currentTrainRow][c]);
                }
                printf(" y: ");
                for (int c=0;c<outputCount;c++)
                {
                    printf("%d ",y_train[currentTrainRow][c]);
                }
                printf(" class: ");
                printf("%d",y_train_class[currentTrainRow]);
                printf("\n");
                currentTrainRow++;
            }
        }

        //Print test data
        currentTestRow=0;
        printf("\n\nTEST:\n");
        for (int i=(currentFold*testStandardSampleCount);i<((currentFold*testStandardSampleCount) + numberOfTestSamples);i++)
        {
            printf("INDEX:%d DATA x: ",indexArray[i]);
            for (int c=0;c<numberOfInputBits;c++)
            {
                printf("%d ",X_test[currentTestRow][c]);
            }
            printf(" y: ");
            for (int c=0;c<outputCount;c++)
            {
                printf("%d ",y_test[currentTestRow][c]);
            }
            printf(" class: ");
            printf("%d",y_test_class[currentTestRow]);
            printf("\n");
            currentTestRow++;
        }
    }
        //K-Fold is over.
        //If evaluation method is selected as train-test split, fill train-test arrays below
    else
    {
        //Determine train-test size
        numberOfTrainSamples = (int)(numberOfTotalRows * train_test_split);
        numberOfTestSamples = numberOfTotalRows - numberOfTrainSamples;

        //Print info
        printf("Number of train samples:");printf("%d",numberOfTrainSamples);printf("\n");
        printf("Number of test samples:");printf("%d",numberOfTestSamples);printf("\n");

        //Shuffle whole dataset by creating a shuffle index
        int indexArray[numberOfTotalRows];
        for (int i=0;i<numberOfTotalRows;i++)
        {
            indexArray[i] = i;
        }
        randomShuffleInt(indexArray,numberOfTotalRows);

        //fill train rows (X)
        for (int x=0;x<numberOfTrainSamples;x++)
        {
            for (int c=0;c<numberOfInputBits;c++)
            {
                X_train[x][c] = X_raw[indexArray[x]][c];
            }
        }

        //fill train rows (y)
        for (int y=0;y<numberOfTrainSamples;y++)
        {
            for (int c=0;c<outputCount;c++)
            {
                y_train[y][c] = y_raw[indexArray[y]][c];
            }
            y_train_class[y] = y_class[indexArray[y]];
        }

        //fill test rows (X)
        for (int x=0;x<numberOfTestSamples;x++)
        {
            for (int c=0;c<numberOfInputBits;c++)
            {
                X_test[x][c] = X_raw[indexArray[numberOfTrainSamples + x]][c];
            }
        }

        //fill test rows (y)
        for (int y=0;y<numberOfTestSamples;y++)
        {
            for (int c=0;c<outputCount;c++)
            {
                y_test[y][c] = y_raw[indexArray[numberOfTrainSamples + y]][c];
            }
            y_test_class[y] = y_class[indexArray[numberOfTrainSamples + y]];
        }

        //Print shuffled dataset (FULL)
        printf("\n");
        printf("Printing Shuffled Dataset\n");
        for (int i=0;i<numberOfTotalRows;i++)
        {
            printf("INDEX:%d DATA x: ",indexArray[i]);
            for (int c=0;c<numberOfInputBits;c++)
            {
                printf("%d ",X_raw[indexArray[i]][c]);
            }
            printf(" y: ");
            for (int c=0;c<outputCount;c++)
            {
                printf("%d ",y_raw[indexArray[i]][c]);
            }
            printf(" class: ");
            printf("%d",y_class[indexArray[i]]);
            printf("\n");
        }

        //Print train data
        printf("\n");
        printf("Printing Shuffled Dataset\n");
        printf("TRAIN:\n");
        for (int i=0;i<numberOfTrainSamples;i++)
        {
            printf("INDEX:%d DATA x: ",indexArray[i]);
            for (int c=0;c<numberOfInputBits;c++)
            {
                printf("%d ",X_train[i][c]);
            }
            printf(" y: ");
            for (int c=0;c<outputCount;c++)
            {
                printf("%d ",y_train[i][c]);
            }
            printf(" class: ");
            printf("%d",y_train_class[i]);
            printf("\n");
        }

        //Print test data
        printf("\n\nTEST:\n");
        for (int i=0;i<numberOfTestSamples;i++)
        {
            printf("INDEX:%d DATA x: ",indexArray[numberOfTrainSamples + i]);
            for (int c=0;c<numberOfInputBits;c++)
            {
                printf("%d ",X_test[i][c]);
            }
            printf(" y: ");
            for (int c=0;c<outputCount;c++)
            {
                printf("%d ",y_test[i][c]);
            }
            printf(" class: ");
            printf("%d",y_test_class[i]);
            printf("\n");
        }
    }



    //Let's Define GA Parameters
    //Determine population size
    int populationSize = 1000;
    //Number of gates for each population
    int gateCount[populationSize];
    //Determine iteration count
    int iterationCount = 70000;
    //Define max number of gates (GA will randomly assign gate count to each population).
    float maxGates = 18.0f;
    //Number of max connections for each gate. Determined randomly (minimum=2)
    int maxConnections = 5;
    //Are we going to use XOR gates (XOR-XNOR)?
    bool useXOR= true;
    int maxGateType;
    if (useXOR) maxGateType=5;else maxGateType=3;

    //Define column strings
    //popBits are connection masks for each gate. 1 denotes a connection and 0 denotes no connection
    bool **popBits[populationSize];
    //popGateType are gate types (AND, OR, XOR etc.)
    int *popGateType[populationSize];

    //Output Source (determines which gates will be selected as multi-class outputs)
    int outputSource[populationSize][outputCount];

    //Define array variable for advanced confusion matrix (NxN N=number of classes)
    int confusionMatrix[populationSize][numberOfClasses][numberOfClasses];

    //Population accuracies (train)
    float trainScore[populationSize];
    float trainAccuracy[populationSize];

    //Tournament Selection Parameter
    float tournamentRate = 0.05f;
    int tournamentCount = (int)((float)populationSize*tournamentRate);

    //Crossover Parameters
    float crossoverRate = 0.75f;
    float gateTypeCrossoverRate = 0.55f;//This is a new parameter checking if gateType will be crossed during crossover (together with the bits)

    //Mutation Parameters
    float mutationRate = 0.1f;
    float gateMutationRate = 0.55f;//A new parameter to mutate gateType

    //Elitism parameter
    float elitismRate = 0.03f;
    int popCountElite = (int)((float)populationSize*elitismRate);

    //Augmentation parameters
    //*****Important Note******
    //Augmentation will not be executed for multi-class classification
    float augmentationRate = 1.00000f;//How the number of gates will increase
    int maxGatesWithAugmentation;//TBD
    float augmentationPopRate = 0.00f;//What portion of population will be replaced with new, augmented networks
    int popCountToBeAugmented = (int)(populationSize * augmentationPopRate);

    //Let's start GA!
    //Create Initial Population
    for (int popIndex=0; popIndex < populationSize; popIndex++)
    {
        //Create random number of gates
        gateCount[popIndex] = (int)maxGates;//random(1, (int)maxGates);
        popBits[popIndex] = new bool*[gateCount[popIndex]];
        popGateType[popIndex] = new int[gateCount[popIndex]];

        //Create -for loop- for each gate
        //Loop starts with gate0 (after input columns)
        //Let's generate gate content
        for (int gateIndex=0; gateIndex < gateCount[popIndex]; gateIndex++)
        {
            //Determine gate type
            // 0:AND
            // 1:OR
            // 2:NAND
            // 3:NOR
            // 4:XOR
            // 5:XNOR
            popGateType[popIndex][gateIndex]= random(0, maxGateType);
            //Let's start randomly filling gate masks
            //Define sub array
            //The length of array is the number of inputs+gate index (gateindex+numberOfInputBits)
            popBits[popIndex][gateIndex]=new bool [gateIndex + numberOfInputBits];
            //Determine connections count randomly
            int connectionsCount = random(2, maxConnections);
            //Fill connection count times True (1)
            if ((gateIndex+numberOfInputBits)<=connectionsCount)
            {
                for (int i=0;i<(numberOfInputBits+gateIndex);i++) popBits[popIndex][gateIndex][i]=true;
            }
                //Fill the rest with false
            else {
                for (int i=0;i<connectionsCount;i++) popBits[popIndex][gateIndex][i]=true;
                for (int i=connectionsCount; i < (gateIndex + numberOfInputBits); i++) popBits[popIndex][gateIndex][i]=false;
            }

            //Now shuffle 1s with 0s - there you have a random mask! We call this Neuroplasticity...
            //Our brain is doing it many times a day :)
            randomShuffle(popBits[popIndex][gateIndex], gateIndex + numberOfInputBits);

        }//End of creating gates (gateIndex)

        //Initialize Output Source
        for (int i=0;i<outputCount;i++)
        {
            outputSource[popIndex][i] = random(0, numberOfInputBits+gateCount[popIndex]-1);
        }
    }//End of popIndex

    //First, we will define new population variables
    bool **popBitsAfterMutation[populationSize];
    int *popGateTypeAfterMutation[populationSize];
    int gateCountAfterMutation[populationSize];
    int outputSourceAfterMutation[populationSize][outputCount];

    float testAccuracyRawOnEachIteration;
    float testAccuracyNetOnEachIteration;

    float bestTestAccuracyRaw=0;
    int bestTestAccuracyRawPopIndex;
    int bestTestAccuracyRawIteration;
    bool bestTestAccRawChanged;

    float bestTestAccuracyNet=0;
    int bestTestAccuracyNetPopIndex;
    int bestTestAccuracyNetIteration;
    bool bestTestAccNetChanged;

    float trainScoreOnEachIteration[iterationCount];
    float testAccuracyOnEachIteration[iterationCount];

    float trainAccuracyOnIterations[iterationCount];
    float testAccNetOnIterations[iterationCount];

    float bestTrainScore = 0.0f;
    int bestTrainScoreIndex;
    int bestTrainScoreIteration;
    bool bestTrainScoreChanged;

    float bestTrainAccuracy;
    int bestTrainAccuracyPopIndex;
    int bestTrainAccuracyIteration;
    bool bestTrainAccuracyChanged;

    //Let's start GA loop
    for (int iterationIndex=0;iterationIndex<iterationCount;iterationIndex++)
    {
        //SITREP for Iteration
        printf("Iteration: %d,\n",iterationIndex);
        printf("Max Gates:%d\n",(int)maxGates);

        //Re-initialize loop (transfer values from the last iteration)
        if (iterationIndex!=0)
        {
            for (int popIndex=0;popIndex<populationSize;popIndex++) {
                gateCount[popIndex] = gateCountAfterMutation[popIndex];
                popBits[popIndex] = new bool *[gateCount[popIndex]];
                popGateType[popIndex] = new int[gateCount[popIndex]];

                for (int gateIndex=0;gateIndex<gateCount[popIndex];gateIndex++)
                {
                    popGateType[popIndex][gateIndex] = popGateTypeAfterMutation[popIndex][gateIndex];
                    popBits[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];
                    for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                        popBits[popIndex][gateIndex][i] = popBitsAfterMutation[popIndex][gateIndex][i];
                }
                for (int i=0;i<outputCount;i++)
                    outputSource[popIndex][i] = outputSourceAfterMutation[popIndex][i];
            }
        }

        //Calculate train score of each population
        float bestTrainScoreInPop=0;
        float bestTrainAccuracyInPop=0;
        for (int popIndex = 0; popIndex < populationSize; popIndex++) {
            int numberOfCorrect = 0;
            int numberOfCorrectNet = 0;

            //Here we define a new variable for NLCv2
            //It is called y_match_count
            //It is a 2-Dimensional matrix (output bits x gates), showing which gate (including the inputs)
            //is the most suitable to be selected as connected to an output bit
            //by statistically collecting matching data
            //We check every row of train data and increment y_match_count of designated cell
            //if it matches with the output data
            int y_match_count[outputCount][numberOfInputBits + gateCount[popIndex]];
            //Zeroize y-match-count
            for (int i=0;i<outputCount;i++) {
                for (int j = 0; j < (numberOfInputBits + gateCount[popIndex]); j++)
                    y_match_count[i][j] = 0;
            }
            //Define bit output of each gate (including inputs)
            bool bitOutput[numberOfTrainSamples][numberOfInputBits + gateCount[popIndex]];
            //Start with iterating through input rows
            for (int rowIndex = 0; rowIndex < numberOfTrainSamples; rowIndex++) {
                //First fill bitOutput with inputs
                for (int colIndex = 0; colIndex < numberOfInputBits; colIndex++) {
                    bitOutput[rowIndex][colIndex] = X_train[rowIndex][colIndex];
                    //Increment dedicated cell if bitOutput matches output
                    for (int yIndex=0;yIndex<outputCount;yIndex++)
                        if (y_train[rowIndex][yIndex]==bitOutput[rowIndex][colIndex]) y_match_count[yIndex][colIndex]++;
                }
                //Then, iterate through gates and calculate gate outputs
                for (int gateIndex = 0; gateIndex < gateCount[popIndex]; gateIndex++) {
                    switch (popGateType[popIndex][gateIndex]) {
                        //AND Gate
                        case 0:
                            bitOutput[rowIndex][numberOfInputBits + gateIndex] = getAndMask(popBits[popIndex][gateIndex], bitOutput[rowIndex],
                                                                                            (gateIndex + numberOfInputBits));
                            break;
                            //OR Gate
                        case 1:
                            bitOutput[rowIndex][numberOfInputBits + gateIndex] = getOrMask(popBits[popIndex][gateIndex], bitOutput[rowIndex],
                                                                                           (gateIndex + numberOfInputBits));
                            break;
                            //NAND Gate
                        case 2:
                            bitOutput[rowIndex][numberOfInputBits + gateIndex] = getNAndMask(popBits[popIndex][gateIndex], bitOutput[rowIndex],
                                                                                             (gateIndex + numberOfInputBits));
                            break;
                            //NOR Gate
                        case 3:
                            bitOutput[rowIndex][numberOfInputBits + gateIndex] = getNOrMask(popBits[popIndex][gateIndex], bitOutput[rowIndex],
                                                                                            (gateIndex + numberOfInputBits));
                            break;
                            //XOR Gate
                        case 4:
                            bitOutput[rowIndex][numberOfInputBits + gateIndex] = getXOrMask(popBits[popIndex][gateIndex], bitOutput[rowIndex],
                                                                                            (gateIndex + numberOfInputBits));
                            break;
                            //XNOR Gate
                        case 5:
                            bitOutput[rowIndex][numberOfInputBits + gateIndex] = getXNOrMask(popBits[popIndex][gateIndex], bitOutput[rowIndex],
                                                                                             (gateIndex + numberOfInputBits));
                            break;
                    }//end of gate switch
                    //Again, increment y_match_count of dedicated cell it bitOutput matches output
                    //This will help us understand which gates are the most suitable to be selected as outputs of our network
                    for (int yIndex=0;yIndex<outputCount;yIndex++) {
                        if (y_train[rowIndex][yIndex] == bitOutput[rowIndex][numberOfInputBits + gateIndex])
                            y_match_count[yIndex][numberOfInputBits + gateIndex]++;
                    }
                }//end loop for gate index

                //This is one of the most critical part of the process.
                //We check each gate (including outputs) and find the max suitable gate
                //to be assigned as output source.
                //Then, to make a simple calculation, we count the matching bits.
                //The number of matching bits is used as the primary reference for calculating GA fitness.
                //We will call it as 'Training score'.

                //The trick here is, we increment the matching bits in a cumulative way.
                //We could do this after completing the entire training rows but moving the step
                //here helps GA to converge faster.
                for (int yIndex=0;yIndex<outputCount;yIndex++) {
                    outputSource[popIndex][yIndex] = findMaxSourceIndex(y_match_count[yIndex],
                                                                        numberOfInputBits + gateCount[popIndex]);
                    numberOfCorrect += findMaxSourceValue(y_match_count[yIndex],
                                                          numberOfInputBits + gateCount[popIndex]);
                }

            }//end loop for Rows. Now we went over all training samples in the dataset.

            //As we determined the output sources (best gates to be selected as outputs)
            //now, we can go over the training samples again and calculate training accuracy.
            //IT should be noted that, training accuracy won't be used as GA fitness.
            //It is just a performance indicator during the algorithm run.
            for (int rX=0;rX<numberOfTrainSamples;rX++) {
                bool rowFullyCorrect = true;
                for (int mIndex = 0; mIndex < outputCount; mIndex++) {
                    if (y_train[rX][mIndex] != bitOutput[rX][outputSource[popIndex][mIndex]]) rowFullyCorrect = false;
                }
                if (iterationIndex==0 && popIndex==0) printf("TRAIN:%d%d OUTPUT:%d%d\n",(int)y_train[rX][0],(int)y_train[rX][1],(int)bitOutput[rX][outputSource[popIndex][0]],(int)bitOutput[rX][outputSource[popIndex][1]]);
                if (rowFullyCorrect) numberOfCorrectNet++;
            }

            //Now, it is time to calculate training score. We divide it to 1000 to make the number smaller. No other reason;
            trainScore[popIndex] = (float) numberOfCorrect / (float) (1000);
            //We can also calculate the training accuracy of this population member
            trainAccuracy[popIndex] = (float) numberOfCorrectNet / (float) (numberOfTrainSamples);

            //Once we get higher scores, we mark it as the new best (of the current iteration).
            if (trainScore[popIndex] > bestTrainScoreInPop)
            {
                bestTrainScoreInPop = trainScore[popIndex];
                bestTrainAccuracyInPop = trainAccuracy[popIndex];
            }

            //In order to monitor progress, we refresh and get the new best in the whole GA run
            if (bestTrainScoreInPop > bestTrainScore)
            {
                bestTrainScore = bestTrainScoreInPop;
                bestTrainScoreIndex = popIndex;
                bestTrainScoreIteration = iterationIndex;
                bestTrainScoreChanged = true;

                bestTrainAccuracy = bestTrainAccuracyInPop;
                bestTrainAccuracyPopIndex = popIndex;
                bestTrainAccuracyIteration = iterationIndex;
                bestTrainAccuracyChanged = true;

                //Once we get the new best, print the network and accuracy details
                if (reportTrainNetwork) {
                    printf("\n*************** Printing Gates for the new best on TRAIN Score **********\n");
                    printf("Iteration: %d\n", bestTrainScoreIteration);
                    printf("Population member: %d\n", bestTrainScoreIndex);
                    printf("Train Score: %f\n", trainScore[bestTrainScoreIndex]);
                    printf("Train accuracy: %f\n", trainAccuracy[bestTrainScoreIndex]);
                    printf("Gate count for best pop: %d\n", gateCount[bestTrainScoreIndex]);
                    printf("--------------------------------------------------\n");
                    for (int gIndex = 0; gIndex < gateCount[bestTrainScoreIndex]; gIndex++) {
                        printf("Gate: %d", gIndex);
                        printf("\n");
                        printf("Gate type: %d\n", popGateType[bestTrainScoreIndex][gIndex]);
                        printf("Gate connections:\n");
                        printArray(popBits[bestTrainScoreIndex][gIndex], gIndex + numberOfInputBits, numberOfInputBits);
                    }
                    printf("*************** End of Printing Gates **********\n");
                    printf("Printing output sources\n");
                    for (int i = 0; i < outputCount; i++)
                        printf("%d ", outputSource[bestTrainScoreIndex][i]);
                    printf("\n-------------------------------\n");
                }
            }

        }//end loop for popIndex
        trainScoreOnEachIteration[iterationIndex] = bestTrainScoreInPop;
        trainAccuracyOnIterations[iterationIndex] = bestTrainAccuracyInPop;

        //Now, we calculate test accuracy in this iteration and report the best among populations
        float bestTestAccInPop=0;
        float bestTestAccNetInPop=0;
        for (int popIndex = 0; popIndex < populationSize; popIndex++) {
            int numberOfCorrect = 0;
            int numberOfCorrectNet = 0;
            //Zeroize confusion matrix
            for (int p=0;p<numberOfClasses;p++)
            {
                for (int p1=0;p1<numberOfClasses;p1++)
                {
                    confusionMatrix[popIndex][p][p1]=0;
                }
            }
            //Define bit output
            bool bitOutput[numberOfInputBits + gateCount[popIndex]];
            //Start with iterating through input rows
            for (int rowIndex = 0; rowIndex < numberOfTestSamples; rowIndex++) {
                for (int colIndex = 0; colIndex < numberOfInputBits; colIndex++) {
                    bitOutput[colIndex] = X_test[rowIndex][colIndex];
                }
                //Then, iterate through gates and calculate gate outputs
                for (int gateIndex = 0; gateIndex < gateCount[popIndex]; gateIndex++) {
                    switch (popGateType[popIndex][gateIndex]) {
                        //AND Gate
                        case 0:
                            bitOutput[numberOfInputBits + gateIndex] = getAndMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                  (gateIndex + numberOfInputBits));
                            break;
                            //OR Gate
                        case 1:
                            bitOutput[numberOfInputBits + gateIndex] = getOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                 (gateIndex + numberOfInputBits));
                            break;
                            //NAND Gate
                        case 2:
                            bitOutput[numberOfInputBits + gateIndex] = getNAndMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                   (gateIndex + numberOfInputBits));
                            break;
                            //NOR Gate
                        case 3:
                            bitOutput[numberOfInputBits + gateIndex] = getNOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                  (gateIndex + numberOfInputBits));
                            break;
                            //XOR Gate
                        case 4:
                            bitOutput[numberOfInputBits + gateIndex] = getXOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                  (gateIndex + numberOfInputBits));
                            break;
                            //XNOR Gate
                        case 5:
                            bitOutput[numberOfInputBits + gateIndex] = getXNOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                   (gateIndex + numberOfInputBits));
                            break;
                    }//end of gate switch
                }//end loop for gate index


                //Increment number of correct prediction if it matches test output
                //This is going to be used to calculate raw accuracy.
                //Raw test accuracy means how many bits in the total output columns matches with the correct ones.
                //This is different from the test accuracy and usually higher than the net accuracy.
                for (int r=0;r<outputCount;r++)
                    if (y_test[rowIndex][r] == bitOutput[outputSource[popIndex][r]]) numberOfCorrect++;

                //Now, we can calculate the net accuracy by checking if all bits match to expected output
                bool rowFullyCorrect = true;
                for (int mIndex=0;mIndex<outputCount;mIndex++)
                {
                    if (y_test[rowIndex][mIndex] != bitOutput[outputSource[popIndex][mIndex]]) rowFullyCorrect= false;
                }
                if (rowFullyCorrect) numberOfCorrectNet++;

                //We populate confusion matrix by filling the 2D array.
                //First, we need to know the category output of our network.
                bool outputBits[outputCount];
                for (int h=0;h<outputCount;h++) outputBits[h] = bitOutput[outputSource[popIndex][h]];
                //Then, fill the relevant cell with the predicted/actual class.
                confusionMatrix[popIndex][getClass(outputBits, outputCount)][y_test_class[rowIndex]]++;

            }//end loop for Rows. Test samples are complete.

            //Now, let's calculate the test accuracy
            float testAcc = (float) numberOfCorrect / (float) (numberOfTestSamples * outputCount);//RAW
            float testAccNet = (float) numberOfCorrectNet / (float) (numberOfTestSamples);//NET

            //Record best test acc in population
            if (testAccNet>bestTestAccNetInPop)
            {
                bestTestAccInPop=testAcc;
                bestTestAccNetInPop = testAccNet;
            }

            //Record best test acc overall
            if (testAccNet > bestTestAccuracyNet)
            {
                bestTestAccuracyRaw = testAcc;
                bestTestAccuracyRawPopIndex = popIndex;
                bestTestAccuracyRawIteration = iterationIndex;
                bestTestAccRawChanged = true;

                bestTestAccuracyNet = testAccNet;
                bestTestAccuracyNetPopIndex = popIndex;
                bestTestAccuracyNetIteration = iterationIndex;
                bestTestAccNetChanged = true;

                //Print the best to monitor progress
                printf("\n*************** Printing Gates for the new best on Test Accuracy **********\n");
                printf("Iteration: %d\n", bestTestAccuracyRawIteration);
                printf("Population member: %d\n", bestTestAccuracyRawPopIndex);
                printf("Train Score: %f\n", trainScore[bestTestAccuracyRawPopIndex]);
                printf("Test accuracy (RAW): %f\n", bestTestAccuracyRaw);
                printf("Train accuracy: %f\n", trainAccuracy[bestTestAccuracyRawPopIndex]);
                printf("Test accuracy (NET): %f\n", bestTestAccuracyNet);
                printf("Gate count for best pop: %d\n",gateCount[bestTestAccuracyRawPopIndex]);
                printf("--------------------------------------------------\n");
                //Print gates (RAW) - Not all gates here has an effect to outcome
                /*for (int gIndex=0;gIndex<gateCount[bestTestAccuracyRawPopIndex]; gIndex++)
                {
                    printf("Gate: %d",gIndex);
                    printf("\n");
                    printf("Gate type: %d\n",popGateType[bestTestAccuracyRawPopIndex][gIndex]);
                    printf("Gate connections:\n");
                    printArray(popBits[bestTestAccuracyRawPopIndex][gIndex], gIndex + numberOfInputBits, numberOfInputBits);
                }
                printf("*************** End of Printing Gates **********\n");
                */
                //Report output sources (Could be either a gate or input. That's why we call it 'source'.)
                printf("Printing output sources\n");
                for (int i=0;i<outputCount;i++)
                    printf("%d ",outputSource[bestTestAccuracyRawPopIndex][i]);
                printf("\n-------------------------------\n");

                printf("********* Report final gates *********\n");
                //First let's find out which gates have direct effect to output
                bool isGateActive[gateCount[bestTestAccuracyRawPopIndex]];

                //Start with the output source (only if they are not direct links from inputs)
                for (int i=0;i<gateCount[bestTestAccuracyRawPopIndex];i++) isGateActive[i]= false;
                for (int i=0;i<outputCount;i++)
                {
                    int outputSourceCurrent = outputSource[bestTestAccuracyRawPopIndex][i];
                    if (outputSourceCurrent>=numberOfInputBits) {
                        isGateActive[outputSourceCurrent-numberOfInputBits]= true;
                        for (int k = numberOfInputBits; k < outputSourceCurrent; k++) {
                            if (popBits[bestTestAccuracyRawPopIndex][outputSourceCurrent-numberOfInputBits][k]) isGateActive[k-numberOfInputBits] = true;
                        }
                    }
                }

                //Now, iteratively check which gates are active when obtaining outputs
                for (int i=gateCount[bestTestAccuracyRawPopIndex]-1;i>=0;i--)
                {
                    bool isGateOutput= false;
                    //We have already covered the output source gates. So let's skip them
                    for (int j=0;j<outputCount;j++)
                    {
                        if (i==(outputSource[bestTestAccuracyRawPopIndex][j]-numberOfInputBits))
                        {
                            isGateOutput= true;
                        }
                    }
                    //Now, mark the active gates
                    if (!isGateOutput && isGateActive[i])
                    {
                        for (int k = 0; k < i; k++) {
                            if (popBits[bestTestAccuracyRawPopIndex][i][k+numberOfInputBits]) isGateActive[k] = true;
                        }
                    }
                }
                //Report only the active gates (if there is none, report will be empty)
                for (int gIndex=0;gIndex<gateCount[bestTestAccuracyRawPopIndex]; gIndex++)
                {
                    if (isGateActive[gIndex]) {
                        printf("Gate: %d", gIndex);
                        printf("\n");
                        printf("Gate type: %d\n", popGateType[bestTestAccuracyRawPopIndex][gIndex]);
                        printf("Gate connections:\n");
                        printArray(popBits[bestTestAccuracyRawPopIndex][gIndex], gIndex + numberOfInputBits,
                                   numberOfInputBits);
                    }
                }
                printf("*************** End of Printing Gates **********\n");


                //It's time to report Confusion matrix
                printf("Confusion Matrix\n");
                for (int i=0;i<numberOfClasses;i++)
                {
                    for (int j=0;j<numberOfClasses;j++)
                    {
                        printf("%d ", confusionMatrix[bestTestAccuracyRawPopIndex][i][j]);
                    }
                    printf("\n");
                }
                printf("\n-------------------------------\n");
                printf("Class Report\n\n");
                for (int i=0;i<numberOfClasses;i++)
                {
                    printf("Class:%d\n",i);
                    int sumHorizontal = 0;
                    for (int a=0;a<numberOfClasses;a++) sumHorizontal+=confusionMatrix[bestTestAccuracyRawPopIndex][i][a];
                    int sumVertical = 0;
                    for (int a=0;a<numberOfClasses;a++) sumVertical+=confusionMatrix[bestTestAccuracyRawPopIndex][a][i];
                    float recall = (float)confusionMatrix[bestTestAccuracyRawPopIndex][i][i] / (float)sumVertical;
                    float precision = (float)confusionMatrix[bestTestAccuracyRawPopIndex][i][i] / (float)sumHorizontal;
                    float f1Score = ((precision*recall)/(precision+recall))*2;
                    printf("Recall (TPR):%f\n",recall);
                    printf("Precision:%f\n",precision);
                    printf("F1-Score:%f\n\n",f1Score);
                }
                printf("\n-------------------------------\n");
            }

        }//end loop for test accuracy
        printf("..Best Test Accuracy (RAW): %f at Iteration %d, Pop %d, with Train Score:%f\n", bestTestAccuracyRaw, bestTestAccuracyRawIteration,
               bestTestAccuracyRawPopIndex, trainScore[bestTestAccuracyRawPopIndex]);
        printf("..Best Test Accuracy (NET): %f at Iteration %d, Pop %d, with Train accuracy:%f\n", bestTestAccuracyNet, bestTestAccuracyNetIteration,
               bestTestAccuracyNetPopIndex, trainAccuracy[bestTestAccuracyRawPopIndex]);
        testAccuracyOnEachIteration[iterationIndex] = bestTestAccInPop;


        //Now we sort the population based on training score
        float tempValue;
        int tempIndex;

        //Define popIndex Original and initialize
        int sortedPopIndex[populationSize];
        for (int i=0;i<populationSize;i++)
            sortedPopIndex[i]=i;//Not yet sorted. We just initialized in ascending order.

        //Do a quick bubble sort and record indexes
        for(int i=0;i<populationSize-1;i++)
        {
            for(int j=0;j<populationSize-i-1;j++)
            {
                if(trainScore[j] > trainScore[j + 1])
                {
                    tempValue=trainScore[j + 1];
                    tempIndex = sortedPopIndex[j + 1];
                    trainScore[j + 1]=trainScore[j];
                    sortedPopIndex[j + 1]=sortedPopIndex[j];
                    trainScore[j]=tempValue;
                    sortedPopIndex[j]=tempIndex;
                }
            }
        }

        //Now move everything to new (sorted) population
        //Let's start with defining new variables for sorted population
        bool **popBitsSorted[populationSize];
        int *popGateTypeSorted[populationSize];
        int gateCountSorted[populationSize];
        int outputSourceSorted[populationSize][outputCount];

        //Ok, now population scores are already sorted, and we have the indexes
        //Now we will form the new population in three steps
        //First we will start with newly created individuals with augmented gates
        //They will fill the slots of the worst population members, coming in the first place

        //We skip this step in NLCv2
        for (int popIndex=0;popIndex<popCountToBeAugmented;popIndex++)
        {

            gateCountSorted[popIndex] = random(1, (int)maxGates);
            popBitsSorted[popIndex] = new bool*[gateCountSorted[popIndex]];
            popGateTypeSorted[popIndex] = new int[gateCountSorted[popIndex]];

            //Let's generate gate content for newly created individuals
            for (int gateIndex=0; gateIndex < gateCountSorted[popIndex]; gateIndex++)
            {
                //Determine gate type
                // 0:AND
                // 1:OR
                // 2:NAND
                // 3:NOR
                // 4:XOR
                // 5:XNOR
                popGateTypeSorted[popIndex][gateIndex]= random(0, maxGateType);

                //Let's start randomly filling gate masks
                //Define sub array
                //The length of array is number of inputs+gate index (gateindex+numberOfInputBits)
                popBitsSorted[popIndex][gateIndex]=new bool [gateIndex + numberOfInputBits];
                //Generate a random number for connections count
                int connectionsCount = random(2, maxConnections);
                //Fill connection times true
                if ((gateIndex+numberOfInputBits)<=connectionsCount)
                {
                    for (int i=0;i<(numberOfInputBits+gateIndex);i++) popBitsSorted[popIndex][gateIndex][i]=true;
                }
                    //Fill the rest with false
                else {
                    for (int i=0;i<connectionsCount;i++) popBitsSorted[popIndex][gateIndex][i]=true;
                    for (int i=connectionsCount; i < (gateIndex + numberOfInputBits); i++) popBitsSorted[popIndex][gateIndex][i]=false;
                }
                //Now shuffle'em! There you have a randomized neural connections.
                randomShuffle(popBitsSorted[popIndex][gateIndex], gateIndex + numberOfInputBits);//Shuffle contents
            }//End of gateIndex

            //Initialize Output Source
            for (int i=0;i<outputCount;i++)
            {
                outputSourceSorted[popIndex][i] = random(0, numberOfInputBits+gateCountSorted[popIndex]-1);
            }
        }

        //Now, calculate training scores of newly introduced population members (wish us luck!)
        //We skip this step in NLCv2
        for (int popIndex = 0; popIndex < popCountToBeAugmented; popIndex++) {

            int numberOfCorrect = 0;
            int y_match_count[outputCount][numberOfInputBits + gateCountSorted[popIndex]];
            //Zeroize y-match-count
            for (int i=0;i<outputCount;i++)
                for (int j=0;j<(numberOfInputBits + gateCountSorted[popIndex]);j++)
                    y_match_count[i][j]=0;
            //Define bit output for each gate (including inputs)
            bool bitOutput[numberOfInputBits + gateCountSorted[popIndex]];
            //Start with iterating through input rows
            for (int rowIndex = 0; rowIndex < numberOfTrainSamples; rowIndex++) {
                //First fill bitOutput with inputs
                for (int colIndex = 0; colIndex < numberOfInputBits; colIndex++) {
                    bitOutput[colIndex] = X_train[rowIndex][colIndex];
                    for (int yIndex=0;yIndex<outputCount;yIndex++)
                        if (y_train[rowIndex][yIndex]==bitOutput[colIndex]) y_match_count[yIndex][colIndex]++;
                }

                //Then, iterate through gates and calculate gate outputs
                for (int gateIndex = 0; gateIndex < gateCountSorted[popIndex]; gateIndex++) {
                    switch (popGateType[popIndex][gateIndex]) {
                        //AND Gate
                        case 0:
                            bitOutput[numberOfInputBits + gateIndex] = getAndMask(popBitsSorted[popIndex][gateIndex],
                                                                                  bitOutput, (gateIndex + numberOfInputBits));
                            break;
                            //OR Gate
                        case 1:
                            bitOutput[numberOfInputBits + gateIndex] = getOrMask(popBitsSorted[popIndex][gateIndex],
                                                                                 bitOutput, (gateIndex + numberOfInputBits));
                            break;
                            //NAND Gate
                        case 2:
                            bitOutput[numberOfInputBits + gateIndex] = getNAndMask(popBitsSorted[popIndex][gateIndex],
                                                                                   bitOutput, (gateIndex + numberOfInputBits));
                            break;
                            break;
                            //NOR Gate
                        case 3:
                            bitOutput[numberOfInputBits + gateIndex] = getNOrMask(popBitsSorted[popIndex][gateIndex],
                                                                                  bitOutput, (gateIndex + numberOfInputBits));
                            break;
                            //XOR Gate
                        case 4:
                            bitOutput[numberOfInputBits + gateIndex] = getXOrMask(popBitsSorted[popIndex][gateIndex],
                                                                                  bitOutput, (gateIndex + numberOfInputBits));
                            break;
                            //XNOR Gate
                        case 5:
                            bitOutput[numberOfInputBits + gateIndex] = getXNOrMask(popBitsSorted[popIndex][gateIndex],
                                                                                   bitOutput, (gateIndex + numberOfInputBits));
                            break;
                    }//end of gate switch
                    for (int yIndex=0;yIndex<outputCount;yIndex++)
                    {
                        if (y_train[rowIndex][yIndex]==bitOutput[numberOfInputBits + gateIndex])
                            y_match_count[yIndex][numberOfInputBits + gateIndex]++;
                    }
                }//end loop for gate index
                //Now, we can check if the last gate (output) is equal to expected output
                //if (bitOutput[numberOfInputBits + gateCount[popIndex] - 1] == y_train[rowIndex][0]) numberOfCorrect++;

            }//end loop for Rows
            //And, here is the result..
            for (int yIndex=0;yIndex<outputCount;yIndex++) {
                outputSourceSorted[popIndex][yIndex] = findMaxSourceIndex(y_match_count[yIndex],
                                                                          numberOfInputBits + gateCountSorted[popIndex]);
                numberOfCorrect+= findMaxSourceValue(y_match_count[yIndex],
                                                     numberOfInputBits + gateCountSorted[popIndex]);
            }
            trainScore[popIndex] = (float) numberOfCorrect / (float) (numberOfTrainSamples * outputCount);
        }//end loop for popIndex (Calculate accuracies of newly introduced population members)


        //Now, move existing population contents to sorted population
        //We skip this step in NLCv2
        for (int popIndex=popCountToBeAugmented;popIndex<populationSize;popIndex++)
        {
            gateCountSorted[popIndex] = gateCount[sortedPopIndex[popIndex]];
            popBitsSorted[popIndex] = new bool*[gateCountSorted[popIndex]];
            popGateTypeSorted[popIndex] = new int[gateCountSorted[popIndex]];

            //Iterate through each gate. Moving is always hard!
            for (int gateIndex=0;gateIndex<gateCountSorted[popIndex];gateIndex++)
            {
                //Define new array for popBits
                popBitsSorted[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];
                //Get gateType from old index
                popGateTypeSorted[popIndex][gateIndex] = popGateType[sortedPopIndex[popIndex]][gateIndex];
                //Now let's move popBits to popBitsSorted
                for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                    popBitsSorted[popIndex][gateIndex][i] = popBits[sortedPopIndex[popIndex]][gateIndex][i];
            }

            for (int r=0;r<outputCount;r++)
                outputSourceSorted[popIndex][r] = outputSource[sortedPopIndex[popIndex]][r];
        }

        //We can start Tournament Selection
        //First, we will create new population variables
        bool **popBitsAfterTournament[populationSize];
        int *popGateTypeAfterTournament[populationSize];
        int gateCountAfterTournament[populationSize];
        int outputSourceAfterTournament[populationSize][outputCount];

        //Iterate populationSize-popCountElite times (Careful: elites can be selected as winners but do not override the elite!)
        for (int popIndex=0;popIndex<populationSize-popCountElite;popIndex++)
        {
            //Define temp array for tournament candidates
            int tempTourIndex[tournamentCount];
            float tempTourAcc[tournamentCount];

            //Pick random members and add to array
            for (int i=0;i<tournamentCount;i++)
            {
                int randomIndex = random(0,populationSize-1);//Elites can also be selected in the tournament
                tempTourIndex[i] = randomIndex;
                tempTourAcc[i] = trainScore[randomIndex];
            }
            //Find max acc among selected tournament candidates
            //Here, a good thing is, if acc of two pop members is the same, then findMax function selects the one with fewer gates (SPARSITY!)
            //And the Oscar goes to....:
            int tournamentWinner = tempTourIndex[findMax(tempTourAcc, tournamentCount, gateCountSorted, tempTourIndex)];

            //Now, fill new population with the winner
            //Initialize variables
            gateCountAfterTournament[popIndex] = gateCountSorted[tournamentWinner];
            popBitsAfterTournament[popIndex] = new bool*[gateCountSorted[tournamentWinner]];
            popGateTypeAfterTournament[popIndex] = new int[gateCountSorted[tournamentWinner]];

            //Move winner to new pop
            //Iterate through each gate..
            for (int gateIndex=0;gateIndex<gateCountAfterTournament[popIndex];gateIndex++)
            {
                //Define new array for popBits
                popBitsAfterTournament[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];
                //Get gateType from old index
                popGateTypeAfterTournament[popIndex][gateIndex] = popGateTypeSorted[tournamentWinner][gateIndex];
                //Now let's move previous popBits to popBitsAfterTournament
                for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                    popBitsAfterTournament[popIndex][gateIndex][i] = popBitsSorted[tournamentWinner][gateIndex][i];
            }

            for (int r=0;r<outputCount;r++)
            {
                outputSourceAfterTournament[popIndex][r] = outputSourceSorted[tournamentWinner][r];
            }

        }//end of tournament iteration

        //Now, tournament selection is over. We can move elites to new population. Red carpet ceremony begins...
        for (int popIndex=populationSize-popCountElite;popIndex<populationSize;popIndex++)
        {
            //Initialize variables
            gateCountAfterTournament[popIndex] = gateCountSorted[popIndex];
            popBitsAfterTournament[popIndex] = new bool*[gateCountSorted[popIndex]];
            popGateTypeAfterTournament[popIndex] = new int[gateCountSorted[popIndex]];

            //Move elites to new pop
            //Iterate through each gate
            for (int gateIndex=0;gateIndex<gateCountAfterTournament[popIndex];gateIndex++)
            {
                //Define new array for popBits
                popBitsAfterTournament[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];
                //Get gateType from old index
                popGateTypeAfterTournament[popIndex][gateIndex] = popGateTypeSorted[popIndex][gateIndex];
                //Now let's move previous popBits to popBitsAfterTournament
                for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                    popBitsAfterTournament[popIndex][gateIndex][i] = popBitsSorted[popIndex][gateIndex][i];
            }

            for (int r=0;r<outputCount;r++)
                outputSourceAfterTournament[popIndex][r] = outputSourceSorted[popIndex][r];

        }

        //Let's start crossover! Nature wants reproduction of species!
        //First, we will create new population variables
        bool **popBitsAfterCrossover[populationSize];
        int *popGateTypeAfterCrossover[populationSize];
        int gateCountAfterCrossover[populationSize];
        int outputSourceAfterCrossover[populationSize][outputCount];

        //Start iterating over each population member and check if it will be crossed or not (we will exclude elites)
        for (int popIndex=0;popIndex<populationSize-popCountElite-1;popIndex++)
        {
            //Generate a random number (to be used for crossover probability)
            double r = ((double) rand() / (RAND_MAX));
            //Check if pop member will be crossed
            if (r<crossoverRate)
            {
                //Determine how many gates will be crossed
                //We will take the first pop member and cross with the consecutive one
                //Half of the first pop will be subject to crossover
                //However, if number of gates of the half of the first pop is fewer than consecutive one, we will only cross gateCount of the consecutive one
                int gateCountA = gateCountAfterTournament[popIndex];
                int gateCountB = gateCountAfterTournament[popIndex+1];

                int halfGateCountA = (int)((gateCountA/2));
                if (halfGateCountA==0) halfGateCountA=1;

                int gateCountToBeCrossed;
                if (halfGateCountA<=gateCountB) gateCountToBeCrossed = halfGateCountA;
                else gateCountToBeCrossed = gateCountB;

                //Initialize variables for new Population
                gateCountAfterCrossover[popIndex] = gateCountAfterTournament[popIndex];
                popBitsAfterCrossover[popIndex] = new bool*[gateCountAfterTournament[popIndex]];
                popGateTypeAfterCrossover[popIndex] = new int[gateCountAfterTournament[popIndex]];

                //Let's move the contents for the gates to be exchanged
                for (int gateIndex=0;gateIndex<gateCountToBeCrossed;gateIndex++)
                {
                    //Initialize array for popBits
                    popBitsAfterCrossover[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                    //Let's decide if gateType will be affected
                    double rForGateType = ((double) rand() / (RAND_MAX));
                    if (rForGateType<gateTypeCrossoverRate)
                    {
                        //Get gateType from consecutive pop
                        popGateTypeAfterCrossover[popIndex][gateIndex] = popGateTypeAfterTournament[popIndex+1][gateIndex];
                    }
                    else {
                        //Get gateType without change
                        popGateTypeAfterCrossover[popIndex][gateIndex] = popGateTypeAfterTournament[popIndex][gateIndex];
                    }

                    //Now let's move popBits to popBits new (move contents of the consecutive pop)
                    for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                        popBitsAfterCrossover[popIndex][gateIndex][i] = popBitsAfterTournament[popIndex+1][gateIndex][i];
                }

                //Let's fill the rest (remaining gates to new population)
                for (int gateIndex=gateCountToBeCrossed;gateIndex<gateCountAfterCrossover[popIndex];gateIndex++)
                {
                    //Initialize array for popBits
                    popBitsAfterCrossover[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                    //Get gateType without change
                    popGateTypeAfterCrossover[popIndex][gateIndex] = popGateTypeAfterTournament[popIndex][gateIndex];

                    //Now let's move popBits to popBits new (no change)
                    for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                        popBitsAfterCrossover[popIndex][gateIndex][i] = popBitsAfterTournament[popIndex][gateIndex][i];
                }

                for (int v=0;v<outputCount;v++)
                    outputSourceAfterCrossover[popIndex][v] = outputSourceAfterTournament[popIndex][v];
            }
            else //No crossover
            {
                //Initialize array
                gateCountAfterCrossover[popIndex] = gateCountAfterTournament[popIndex];
                popBitsAfterCrossover[popIndex] = new bool*[gateCountAfterTournament[popIndex]];
                popGateTypeAfterCrossover[popIndex] = new int[gateCountAfterTournament[popIndex]];

                //Let's move the contents without any change (all gates)
                for (int gateIndex=0;gateIndex<gateCountAfterCrossover[popIndex];gateIndex++)
                {
                    //Initialize array for popBits
                    popBitsAfterCrossover[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                    //Get gateType without change
                    popGateTypeAfterCrossover[popIndex][gateIndex] = popGateTypeAfterTournament[popIndex][gateIndex];

                    //Now let's move popBits to popBits new (directly)
                    for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                        popBitsAfterCrossover[popIndex][gateIndex][i] = popBitsAfterTournament[popIndex][gateIndex][i];
                }

                for (int v=0;v<outputCount;v++)
                    outputSourceAfterCrossover[popIndex][v] = outputSourceAfterTournament[popIndex][v];
            }//End of no crossover condition
        }//end popIndex for Crossover Operation

        //After crossover, move last pop and elite pops directly
        for (int popIndex=populationSize-popCountElite-1;popIndex<populationSize;popIndex++)
        {
            //Initialize variables for new Population
            gateCountAfterCrossover[popIndex] = gateCountAfterTournament[popIndex];
            popBitsAfterCrossover[popIndex] = new bool*[gateCountAfterTournament[popIndex]];
            popGateTypeAfterCrossover[popIndex] = new int[gateCountAfterTournament[popIndex]];

            //Let's move the contents without any change
            for (int gateIndex=0;gateIndex<gateCountAfterCrossover[popIndex];gateIndex++)
            {
                //Initialize array for popBits
                popBitsAfterCrossover[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                //Get gateType without change
                popGateTypeAfterCrossover[popIndex][gateIndex] = popGateTypeAfterTournament[popIndex][gateIndex];

                //Now let's move popBits to popBits new (directly)
                for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                    popBitsAfterCrossover[popIndex][gateIndex][i] = popBitsAfterTournament[popIndex][gateIndex][i];
            }

            for (int r=0;r<outputCount;r++)
                outputSourceAfterCrossover[popIndex][r] = outputSourceAfterTournament[popIndex][r];
        }


        //Let's start mutation! (Sounds like a bad thing but it's good for the species...)

        //Start iterating over each population member and check if it will be mutated or not (we will exclude elites)
        for (int popIndex=0;popIndex<populationSize-popCountElite;popIndex++)
        {

            //Initialize variables for new Population (after mutation variables)
            gateCountAfterMutation[popIndex] = gateCountAfterCrossover[popIndex];
            popBitsAfterMutation[popIndex] = new bool*[gateCountAfterMutation[popIndex]];
            popGateTypeAfterMutation[popIndex] = new int[gateCountAfterMutation[popIndex]];

            //Generate a random number (to be used for mutation probability)
            double r = ((double) rand() / (RAND_MAX));
            //Check if pop member will be mutated
            if (r<mutationRate)
            {
                //Let's iterate through gates and apply mutation if needed
                for (int gateIndex=0;gateIndex<gateCountAfterMutation[popIndex];gateIndex++)
                {
                    //Randomly select gates for mutation
                    double rForMutation = ((double) rand() / (RAND_MAX));
                    if (rForMutation<gateMutationRate)
                    {
                        //Randomly assign a new gate type
                        popGateTypeAfterMutation[popIndex][gateIndex] = random(0,maxGateType);

                        //Fill the bits randomly
                        //Let's start randomly filling gate masks
                        //The length of array is number of inputs+gate index (gateindex+numberOfInputBits)
                        popBitsAfterMutation[popIndex][gateIndex]=new bool [gateIndex + numberOfInputBits];
                        //Generate a random number for connections count
                        int connectionsCount = random(2, maxConnections);
                        //Fill connection times true
                        if ((gateIndex+numberOfInputBits)<=connectionsCount)
                        {
                            for (int i=0;i<(numberOfInputBits+gateIndex);i++) popBitsAfterMutation[popIndex][gateIndex][i]=true;
                        }
                            //Fill the rest with false
                        else {
                            for (int i=0;i<connectionsCount;i++) popBitsAfterMutation[popIndex][gateIndex][i]=true;
                            for (int i=connectionsCount; i < (gateIndex + numberOfInputBits); i++) popBitsAfterMutation[popIndex][gateIndex][i]=false;
                        }
                        //Now, shuffle'em! There you have a mutated gate with random connections...
                        randomShuffle(popBitsAfterMutation[popIndex][gateIndex], gateIndex + numberOfInputBits);//Shuffle contents
                    }//End mutate gate
                    else //Directly move gate (no mutation)
                    {
                        popBitsAfterMutation[popIndex][gateIndex]=new bool [gateIndex + numberOfInputBits];
                        //Get gateType without change
                        popGateTypeAfterMutation[popIndex][gateIndex] = popGateTypeAfterCrossover[popIndex][gateIndex];
                        //Now let's move popBits to popBits new (after mutation)
                        for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                            popBitsAfterMutation[popIndex][gateIndex][i] = popBitsAfterCrossover[popIndex][gateIndex][i];
                    }
                }//end gateindex

                for (int v=0;v<outputCount;v++)
                    outputSourceAfterMutation[popIndex][v] = outputSourceAfterCrossover[popIndex][v];
            }
            else //No Mutation
            {
                //Let's move the contents without any change (all gates)
                for (int gateIndex=0;gateIndex<gateCountAfterMutation[popIndex];gateIndex++)
                {
                    //Initialize array for popBits
                    popBitsAfterMutation[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                    //Get gateType without change
                    popGateTypeAfterMutation[popIndex][gateIndex] = popGateTypeAfterCrossover[popIndex][gateIndex];

                    //Now let's move popBits to popBits new (directly)
                    for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                        popBitsAfterMutation[popIndex][gateIndex][i] = popBitsAfterCrossover[popIndex][gateIndex][i];

                }
                for (int v=0;v<outputCount;v++)
                    outputSourceAfterMutation[popIndex][v] = outputSourceAfterCrossover[popIndex][v];
            }//End of no mutation condition
        }//end popIndex for Mutation Operation

        //After mutation, move elite pops directly (keep them secure!)
        for (int popIndex=populationSize-popCountElite;popIndex<populationSize;popIndex++)
        {
            //Initialize variables for new Population
            gateCountAfterMutation[popIndex] = gateCountAfterCrossover[popIndex];
            popBitsAfterMutation[popIndex] = new bool*[gateCountAfterMutation[popIndex]];
            popGateTypeAfterMutation[popIndex] = new int[gateCountAfterMutation[popIndex]];

            //Let's move the contents without any change
            for (int gateIndex=0;gateIndex<gateCountAfterMutation[popIndex];gateIndex++)
            {
                //Initialize array for popBits
                popBitsAfterMutation[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                //Get gateType without change
                popGateTypeAfterMutation[popIndex][gateIndex] = popGateTypeAfterCrossover[popIndex][gateIndex];

                //Now let's move popBits to popBits new (directly)
                for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                    popBitsAfterMutation[popIndex][gateIndex][i] = popBitsAfterCrossover[popIndex][gateIndex][i];
            }
            for (int r=0;r<outputCount;r++)
                outputSourceAfterMutation[popIndex][r] = outputSourceAfterCrossover[popIndex][r];
        }

    }//end loop for GA iterations

    //We come to the end of GA loop. Now, we have the valuable chromosomes stored at popBitsAfterMutation.
    //If not terminated, we apply the same process for these chromosomes again and we will obtain a new generation
    //Every generation is expected to be better than the previous one!
    //Now, let's see the results...

    printf("\n");

    //Print Overall result
    printf("\n*****************SUMMARY*****************\n");
    printf("Population Count: %d\n",populationSize);
    printf("Tournament ratio: %.2f\n",tournamentRate);
    printf("Probability of crossover: %.2f\n",crossoverRate);
    printf("Probability of mutation: %.2f\n", mutationRate);
    printf("Elitism ratio: %.2f\n",elitismRate);
    printf("Max Connections: %d\n", maxConnections);
    printf("Use XOR: ");if (useXOR) printf("Yes\n");else printf("No\n");
    printf("Use kFold? ");if (isKFold) printf("Yes\n");else printf("No\n");
    if (isKFold) printf("Number of folds:%d\n",numberOfK);
    if (isKFold) printf("Current fold:%d\n",currentFold);
    if (!isKFold) printf("Train-test split: %.2f\n",train_test_split);
    printf("Seed: %d\n",seed);

    printf("\n*************** OVERALL RESULT **********\n");
    printf("Best found on iteration: %d\n", bestTestAccuracyRawIteration);
    printf("Population member: %d\n", bestTestAccuracyRawPopIndex);
    printf("Train score (higher is better): %f\n", trainScore[bestTestAccuracyRawPopIndex]);
    printf("Test accuracy: %f\n", bestTestAccuracyRaw);
    printf("Gate count for best pop: %d\n",gateCount[bestTestAccuracyRawPopIndex]);
    printf("--------------------------------------------------\n");
    for (int gIndex=0;gIndex<gateCount[bestTestAccuracyRawPopIndex]; gIndex++)
    {
        printf("Gate: %d",gIndex);
        if (gIndex== gateCount[bestTestAccuracyRawPopIndex] - 1) printf(" (Output gate)");
        printf("\n");
        printf("Gate type: %d\n",popGateType[bestTestAccuracyRawPopIndex][gIndex]);
        printf("Gate connections:\n");
        printArray(popBits[bestTestAccuracyRawPopIndex][gIndex], gIndex + numberOfInputBits, numberOfInputBits);
    }
    printf("\n****************************** Test Acc History *****************************\n");
    for (int accH=0;accH<iterationCount;accH++)
        printf("%.12f,", testAccuracyOnEachIteration[accH]);
    printf("\n****************************** Train Score History *****************************\n");
    for (int accH=0;accH<iterationCount;accH++)
        printf("%.12f,", trainScoreOnEachIteration[accH]);


    return 0;
}

//Here are some useful functions used in the code.

//Generates a random integer
int random(int from, int to){

    return rand() % (to - from + 1) + from;
}

//Finds the max value inside an array
int findMax(float arr[], int n, int gateCount[], int popIndex[])
{
    float max=0;
    int maxIndex=0;
    for (int i=0;i<n;i++)
    {
        if (arr[i]>max) {
            max=arr[i];
            maxIndex=i;
        }
        else if (arr[i]==max)
        {
            if (gateCount[popIndex[i]]<gateCount[popIndex[maxIndex]])
            {
                maxIndex=i;
            }
        }
    }
    return maxIndex;
}

//Logical OR gate with a given connections mask
bool getOrMask(bool maskArray[], bool outputArray[], int n)
{
    bool result = false;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && outputArray[i]) result= true;
    }
    return result;
}

//Logical NOR gate with a given connections mask
bool getNOrMask(bool maskArray[], bool outputArray[], int n)
{
    bool result = false;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && outputArray[i]) result= true;
    }
    return !result;
}

//Logical AND gate with a given connections mask
bool getAndMask(bool maskArray[], bool outputArray[], int n)
{
    bool result = true;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && !outputArray[i]) result= false;
    }
    return result;
}

//Logical NAND gate with a given connections mask
bool getNAndMask(bool maskArray[], bool outputArray[], int n)
{
    bool result = true;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && !outputArray[i]) result= false;
    }
    return !result;
}

//Logical XOR gate with a given connections mask
bool getXOrMask(bool maskArray[], bool outputArray[], int n)
{
    int numberOfTrue=0;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && outputArray[i]) numberOfTrue++;
    }
    if ((numberOfTrue % 2)==0) return false;
    else return true;
}

//Logical XNOR gate with a given connections mask
bool getXNOrMask(bool maskArray[], bool outputArray[], int n)
{
    int numberOfTrue=0;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && outputArray[i]) numberOfTrue++;
    }
    if ((numberOfTrue % 2)==0) return true;
    else return false;
}

//Swap two boolean values in memory
void swap (bool *a, bool *b)
{
    bool temp = *a;
    *a = *b;
    *b = temp;
}

//Same for an integer
void swapInt (int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

//Shuffle gate connections
void randomShuffle (bool arr[], int n)
{
    // Use a different seed value so that
    // we don't get same r esult each time
    // we run this program

    // Start from the last element and swap
    // one by one. We don't need to run for
    // the first element that's why i > 0
    for (int i = n - 1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        int j = rand() % (i + 1);

        // Swap arr[i] with the element
        // at random index
        swap(&arr[i], &arr[j]);
    }
}

//Shuffle integers
void randomShuffleInt (int arr[], int n)
{

    // Start from the last element and swap
    // one by one. We don't need to run for
    // the first element that's why i > 0
    for (int i = n - 1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        int j = rand() % (i + 1);

        // Swap arr[i] with the element
        // at random index
        swapInt(&arr[i], &arr[j]);
    }
}

//Prints gate connections
void printArray (bool arr[], int n, int numberofinputbits)
{
    for (int i = 0; i < n; i++)
        if (i<numberofinputbits) printf("g%d:%d ",i, arr[i]);
        else printf("g%d:%d ",i-numberofinputbits, arr[i]);
    printf("\n");
}

//Converts boolean to string (true or false)
inline const char * const BoolToString(bool b)
{
    return b ? "true" : "false";
}

int findMaxSourceIndex(int inputArray[], int arrayLength)
{
    int max=0;
    int index=0;
    for (int i=0;i<arrayLength;i++)
        if (inputArray[i]>max) {
            max=inputArray[i];
            index=i;
        }
    return index;
}

int findMaxSourceValue(int inputArray[], int arrayLength)
{
    int max=0;
    int index=0;
    for (int i=0;i<arrayLength;i++)
        if (inputArray[i]>max) {
            max=inputArray[i];
            index=i;
        }
    return max;
}

int roundP(float d)
{
    return (int)(floor(d + 0.5));
}
int powerF(int x , int y )
{
    if(y==0)
        return 1;
    return (x*powerF(x,y-1));
}

int getClass(bool arr[], int n)
{
    int result=0;
    for (int i=0;i<n;i++)
    {
        result+=arr[i]*(powerF(2,(n-1-i)));
    }
    return result;
}

