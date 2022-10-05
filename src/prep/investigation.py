"""
Description:
    The script investigation code related to
    the NBI and NBE dataset

Author: Akshay Kale
Date: July 16th, 2021
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
    for record in records:
        structureNumber = record[1]
        value = dictionary.get(structureNumber)
        record.append(value)
        newRecords.append(record)
    header.append(name)
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
    pathNbe = "../../data/nbe_processed/nbe.csv"
    pathNbi =  "nebraska_deep_learning.csv"

    listOfNbeRecords, nbeHeader = read_csv(pathNbe)
    listOfNbiRecords, nbiHeader = read_csv(pathNbi)

    # Create a dictionary of the NBI and
    structNumsNBI = list()
    for record in listOfNbiRecords:
        structureNumber = record[1]
        structNumsNBI.append(structureNumber)

    structNumsNBE = list()
    for record in listOfNbeRecords:
        structureNumber = record[2]
        structNumsNBE.append(structureNumber)

    structNumsNBI =  set(structNumsNBI)
    structNumsNBE =  set(structNumsNBE)
    if (structNumsNBI & structNumsNBE):
        print(structNumsNBI & structNumsNBE)
    else:
        print('No common elements')

    # How many bridges with perfCat:
    uniqueBridges = set()
    for record in listOfNbeRecords:
        structureNumber = record[2]
        state = record[1]
        category = record[9]
        if category !="" and state == "31":
           uniqueBridges.add(structureNumber)
    print(len(uniqueBridges))
    #nbeDict = group_by(listOfNbeRecords)
    #structureQuantityDict, structureMajorElement = compute_attributes(nbeDict)


if __name__=='__main__':
    main()
