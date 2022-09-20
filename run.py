import os
from model import testing,evaluate

train_person_id = input("Enter person's id : ")
test_image_path = input("Enter person image path: ")

 
train_path = 'C:/Users/Dell/check/Dataset/Features/Training/training_'+train_person_id+'.csv'
testing(test_image_path)
test_path = 'C:/Users/Dell/check/Dataset/TestFeatures/testcsv.csv'

evaluate(train_path, test_path, type2=True)    