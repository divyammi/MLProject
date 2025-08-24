import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train_data.csv")
    test_data_path: str = os.path.join('artifacts', "test_data.csv")
    raw_data_path: str = os.path.join('artifacts', "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion mehtod")
        try:
            df = pd.read_csv("C:/Users/divyamm/Downloads/Learning/Python_with_Data_Science/VS_Code/Day48(MLProject)/notebook/data/stud.csv")
            logging.info("Reading dataset from csv as dataframe ")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            logging.info("Train test split initiated")

            train_set,test_set= train_test_split(df,test_size=0.2,random_state=42)

            logging.info("Storing train and test data in respective files")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info("iNJESTION OF DATA COMPLETED")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            pass

if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()






























# print(df.head())
# print(df.shape)

# # check for missing values
# print(df.isna().sum())

# #check for duplicates
# print(df.duplicated().sum())

# #check for data types
# print(df.info())

# #check for unique values in each column
# print(df.nunique())

# #check for statistics of the data set
# print(df.describe())

# #categorical different values
# print("Categores of gender {", df.gender.unique(), "}")
# print("Categores of race_ethnicity {", df.race_ethnicity.unique(), "}")
# print("Categores of parental_level_of_education {", df.parental_level_of_education.unique(), "}")
# print("Categores of lunch {", df.lunch.unique(), "}")


# #define numerical and categorical adta
# num_features = [feature for feature in df.columns if df[feature].dtype!='O']
# cat_features = [feature for feature in df.columns if df[feature].dtype=='O']

# print("Categorical features = " , cat_features, "numerical features = ", num_features)


# # adding more usefule colms
# df['Total_score'] = df['math_score']+ df['reading_score']+ df['writing_score']
# df['Average'] = df["Total_score"] /3

# print(df.head(5))


# #visulaization of data
# fig, axs = plt.subplots(1, 2, figsize=(15, 7))
# plt.subplot(121)
# sns.histplot(data=df,x='Average',bins=50,kde=True,color='g')
# plt.subplot(122)
# sns.histplot(data=df,x='Average',kde=True,hue='gender')
# plt.show()
# #This means female students are performing better in exams




# plt.subplots(1,3,figsize=(25,6))
# plt.subplot(141)
# sns.histplot(data=df,x='average',kde=True,hue='lunch')
# plt.subplot(142)
# sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='lunch')
# plt.subplot(143)
# sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='lunch')
# plt.show()


# '''  Insights
# - Standard lunch helps perform well in exams.
# - Standard lunch helps perform well in exams be it a male or a female.'''



# plt.subplots(1,3,figsize=(25,6))
# plt.subplot(141)
# ax =sns.histplot(data=df,x='average',kde=True,hue='parental_level_of_education')
# plt.subplot(142)
# ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='parental_level_of_education')
# plt.subplot(143)
# ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='parental_level_of_education')
# plt.show()

# ''' Insights
# - In general parent's education don't help student perform well in exam.
# - 2nd plot shows that parent's whose education is of associate's degree or master's degree their male child tend to perform well in exam
# - 3rd plot we can see there is no effect of parent's education on female students.'''



# plt.subplots(1,3,figsize=(25,6))
# plt.subplot(141)
# ax =sns.histplot(data=df,x='average',kde=True,hue='race_ethnicity')
# plt.subplot(142)
# ax =sns.histplot(data=df[df.gender=='female'],x='average',kde=True,hue='race_ethnicity')
# plt.subplot(143)
# ax =sns.histplot(data=df[df.gender=='male'],x='average',kde=True,hue='race_ethnicity')
# plt.show()

# '''Insights
# - Students of group A and group B tends to perform poorly in exam.
# - Students of group A and group B tends to perform poorly in exam irrespective of whether they are male or female'''


# plt.rcParams['figure.figsize'] = (30, 12)

# plt.subplot(1, 5, 1)
# size = df['gender'].value_counts()
# labels = 'Female', 'Male'
# color = ['red','green']


# plt.pie(size, colors = color, labels = labels,autopct = '.%2f%%')
# plt.title('Gender', fontsize = 20)
# plt.axis('off')



# plt.subplot(1, 5, 2)
# size = df['race_ethnicity'].value_counts()
# labels = 'Group C', 'Group D','Group B','Group E','Group A'
# color = ['red', 'green', 'blue', 'cyan','orange']

# plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
# plt.title('Race/Ethnicity', fontsize = 20)
# plt.axis('off')



# plt.subplot(1, 5, 3)
# size = df['lunch'].value_counts()
# labels = 'Standard', 'Free'
# color = ['red','green']

# plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
# plt.title('Lunch', fontsize = 20)
# plt.axis('off')


# plt.subplot(1, 5, 4)
# size = df['test_preparation_course'].value_counts()
# labels = 'None', 'Completed'
# color = ['red','green']

# plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
# plt.title('Test Course', fontsize = 20)
# plt.axis('off')


# plt.subplot(1, 5, 5)
# size = df['parental_level_of_education'].value_counts()
# labels = 'Some College', "Associate's Degree",'High School','Some High School',"Bachelor's Degree","Master's Degree"
# color = ['red', 'green', 'blue', 'cyan','orange','grey']

# plt.pie(size, colors = color,labels = labels,autopct = '.%2f%%')
# plt.title('Parental Education', fontsize = 20)
# plt.axis('off')


# plt.tight_layout()
# plt.grid()

# plt.show()


# '''Insights
# - Number of Male and Female students is almost equal
# - Number students are greatest in Group C
# - Number of students who have standard lunch are greater
# - Number of students who have not enrolled in any test preparation course is greater
# - Number of students whose parental education is "Some College" is greater followed closely by "Associate's Degree"'''




# f,ax=plt.subplots(1,2,figsize=(20,10))
# sns.countplot(x=df['race_ethnicity'],data=df,palette = 'bright',ax=ax[0],saturation=0.95)
# for container in ax[0].containers:
#     ax[0].bar_label(container,color='black',size=20)
    
# plt.pie(x = df['race_ethnicity'].value_counts(),labels=df['race_ethnicity'].value_counts().index,explode=[0.1,0,0,0,0],autopct='%1.1f%%',shadow=True)
# plt.show()   


# '''Insights 
# - Most of the student belonging from group C /group D.
# - Lowest number of students belong to groupA.'''




# #### BIVARIATE ANALYSIS ( Is Race/Ehnicity has any impact on student's performance ? )

# Group_data2=df.groupby('race_ethnicity')
# f,ax=plt.subplots(1,3,figsize=(20,8))
# sns.barplot(x=Group_data2['math_score'].mean().index,y=Group_data2['math_score'].mean().values,palette = 'mako',ax=ax[0])
# ax[0].set_title('Math score',color='#005ce6',size=20)

# for container in ax[0].containers:
#     ax[0].bar_label(container,color='black',size=15)

# sns.barplot(x=Group_data2['reading_score'].mean().index,y=Group_data2['reading_score'].mean().values,palette = 'flare',ax=ax[1])
# ax[1].set_title('Reading score',color='#005ce6',size=20)

# for container in ax[1].containers:
#     ax[1].bar_label(container,color='black',size=15)

# sns.barplot(x=Group_data2['writing_score'].mean().index,y=Group_data2['writing_score'].mean().values,palette = 'coolwarm',ax=ax[2])
# ax[2].set_title('Writing score',color='#005ce6',size=20)

# for container in ax[2].containers:
#     ax[2].bar_label(container,color='black',size=15)


# '''Insights 
# - Group E students have scored the highest marks. 
# - Group A students have scored the lowest marks. 
# - Students from a lower Socioeconomic status have a lower avg in all course subjects'''


# '''Conclusions
# - Student's Performance is related with lunch, race, parental level education
# - Females lead in pass percentage and also are top-scorers
# - Student's Performance is not much related with test preparation course
# - Finishing preparation course is benefitial.'''























































