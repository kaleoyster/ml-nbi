"""
Description:
    Script to create final dataset for deep learning algorithm.
    The script combines the NBI and NBE dataset

Author: Akshay Kale
Date: July 16th, 2021

TODO:
    1. Read nbe dataset [Done]
    2. Read nbi dataset [Done]
    3. Compute the nbe custom variable [Done]
"""
import os
import csv
from tqdm import tqdm
from collections import defaultdict

def read_csv(path):
    """
    Description:
        Read csv file and return a list of records
    """
    listOfRecords = list()
    with open(path, "r") as csvFile:
        csvReader = csv.reader(csvFile,
                              delimiter=",")
        header = next(csvReader)
        for record in csvReader:
            listOfRecords.append(record)
    return listOfRecords, header

def create_dictionary(records, header, key, attribute):
    """
    Description:
      Create the attribute dictionary
    """
    newDictionary = defaultdict()
    for record in records:
       tempDict = dict(zip(header, record))
       structNo = tempDict[key]
       attrVal = tempDict[attribute]
       newDictionary[structNo] = attrVal
    return newDictionary

def group_by(records):
    """
    Description:
       Group all the records by structure number and return a dict
    """
    groupedRecords = defaultdict(list)
    for record in tqdm(records, desc="Grouping records"):
        structureNumber = record[2]
        groupedRecords[structureNumber].append(record)
    return groupedRecords

def compute_majority_element(values):
    """
    Description:
        Computes the majority of elements
    """
    majorityClassDict = defaultdict()
    for value in values:
        qty = value[3]
        element = value[4]
        majorityClassDict[element] = qty
    maxElement = max(zip(majorityClassDict.values(),
                     majorityClassDict.keys()))
    maxElement = int(maxElement[1])
    return maxElement

def compute_total_elements(values):
    """
    Description:
        Computes the majority of elements,
        and reutrn total qty of elements in
        bridge
    """
    totalQty = 0
    majorityClassDict = defaultdict()
    for value in values:
        qty = value[3]
        totalQty = totalQty + int(qty)
    return totalQty

def integrate_attribute(records, header, dictionary, name):
    """
    Description:
        For every bridge, integrate attributes
    """
    newRecords = list()
    counter = 0
    for record in records:
        structureNumber = record[1]
        value = dictionary.get(structureNumber)
        if value == None:
            counter += 1
        record.append(value)
        newRecords.append(record)
    header.append(name)
    print('\nPrinting Counters')
    print(counter)
    return newRecords, header

def compute_attributes(groupedRecords):
    """
    Description:
        For every bridge, compute majority element and
        total quantity of the elements
    """
    structureQuantityDict = defaultdict()
    structureMajorElement = defaultdict()
    for key, values in zip(groupedRecords.keys(), groupedRecords.values()):
        structureName = key
        maxElement = compute_majority_element(values)
        totalQty = compute_total_elements(values)
        structureQuantityDict[key] = totalQty
        structureMajorElement[key] = maxElement
    return structureQuantityDict, structureMajorElement

# Driver function
def main():
    # read csv data
    pathNbe = "../../../data/nbe_processed/nbe.csv"
    pathNbi ="../data/nebraska_deep_learning.csv"

    listOfNbeRecords, nbeHeader = read_csv(pathNbe)
    listOfNbiRecords, nbiHeader = read_csv(pathNbi)

    nbeDict = group_by(listOfNbeRecords)
    structureQuantityDict, structureMajorElement = compute_attributes(nbeDict)
    materialDict = create_dictionary(listOfNbiRecords,
                                      nbiHeader,
                                     'structureNumber',
                                     'material')

    #listOfNbiRecords, nbiHeader = integrate_attribute(listOfNbiRecords,
    #                                                  nbiHeader,
    #                                                  structureMajorElement,
    #                                                  'majorElement')

    #listOfNbiRecords, nbiHeader = integrate_attribute(listOfNbiRecords,
    #                                                  nbiHeader,
    #                                                  structureQuantityDict,
    #                                                  'totalNoElement')

    listOfNbeRecords, nbeHeader = integrate_attribute(listOfNbeRecords,
                                                        nbeHeader,
                                                        materialDict,
                                                        'material')
    # TODO:
    #print("Printing updated records")
    #print(listOfNbeRecords)
    # Why are there no structure Numbers from Nebraska NBI in the NBE records?


if __name__=='__main__':
    main()
