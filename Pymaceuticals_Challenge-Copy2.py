#!/usr/bin/env python
# coding: utf-8

# # Pymaceuticals Inc.
# ---
# 
# ### Analysis
# 
# After running analysis on the data provided, there are a few correlations that can be made. 
# The volume of the tumor is strongly correlated to the weight of the mouse. 
# The drugs Capomulin and Ramicane have a lower mean tumor volume than other drugs that are used. 
# The tumor volume for the mouse selected for Capomulin decreases as the timepoints increased. 
# 

# In[2]:


# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

# Study data files
mouse_metadata_path = "data/Mouse_metadata.csv"
study_results_path = "data/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)
print(mouse_metadata.shape)
print(study_results.shape)

# Combine the data into a single dataset
study_data = pd.merge(study_results, mouse_metadata, on="Mouse ID")
study_data

# Display the data table for preview


# In[3]:


# Checking the number of mice.
unique_mice = study_data['Mouse ID'].unique()
print("Unique mice: " + str(len(unique_mice)))


# In[4]:


# Getting the duplicate mice by ID number that shows up for Mouse ID and Timepoint. 
duplicates = study_data[study_data.duplicated(subset=['Mouse ID', 'Timepoint'])]
duplicate_mice = duplicates['Mouse ID'].unique()
duplicates


# In[4]:


# Optional: Get all the data for the duplicate mouse ID. 


# In[6]:


# Create a clean DataFrame by dropping the duplicate mouse by its ID.
clean_data = study_data.drop(study_data[study_data["Mouse ID"] == "g989"].index)
clean_data


# In[7]:


# Checking the number of mice in the clean DataFrame.
unique_mice = clean_data['Mouse ID'].unique()
print("Unique mice: " + str(len(unique_mice)))


# ## Summary Statistics

# In[8]:


# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen
regimens = clean_data['Drug Regimen'].unique()

# Use groupby and summary statistical methods to calculate the following properties of each drug regimen: 
# mean, median, variance, standard deviation, and SEM of the tumor volume. 
means = clean_data.groupby('Drug Regimen')['Tumor Volume (mm3)'].mean().rename("Mean")
medians = clean_data.groupby('Drug Regimen')['Tumor Volume (mm3)'].median().rename("Median")
variances = clean_data.groupby('Drug Regimen')['Tumor Volume (mm3)'].var().rename("Variance")
stds = clean_data.groupby('Drug Regimen')['Tumor Volume (mm3)'].std().rename("St Dev")
sems = clean_data.groupby('Drug Regimen')['Tumor Volume (mm3)'].sem().rename("SEM")
# Assemble the resulting series into a single summary DataFrame.
summary_stats_df = pd.DataFrame(data=[means, medians, variances, stds, sems]).transpose()
summary_stats_df


# In[9]:


# Generate a summary statistics table of mean, median, variance, standard deviation, 
# and SEM of the tumor volume for each regimen

# Using the aggregation method, produce the same summary statistics in a single line.
agg_stats = clean_data.groupby('Drug Regimen').agg({'Tumor Volume (mm3)': ['mean', 'median', 'var', 'std', 'sem']})
agg_stats


# ## Bar and Pie Charts

# In[10]:


# Generate a bar plot showing the total number of timepoints for all mice tested for each drug regimen using Pandas.
mouse_numbers = clean_data.groupby('Drug Regimen')['Timepoint'].count()
mouse_numbers.plot(kind="bar", x="Drug Regimen", y="Timepoint", width=0.6, figsize=(10,8))
plt.xlabel("Drug Regimen")
plt.ylabel("Number of timepoints (all mice)")
plt.title("Number of timepoints per drug regimen")
plt.ylim(0,250)
plt.show


# In[11]:


# Generate a bar plot showing the total number of timepoints for all mice tested for each drug regimen using pyplot.
x_axis = mouse_numbers.index.tolist()
y_axis = mouse_numbers.tolist()
plt.figure(figsize=(10,8))
plt.bar(x_axis, y_axis, width=0.6)
plt.xticks(rotation=90)
plt.xlabel("Drug Regimen")
plt.ylabel("Number of timepoints (all mice)")
plt.title("Number of timepoints per drug regimen")
plt.ylim(0,250)
plt.show


# In[16]:


# Generate a pie plot showing the distribution of female versus male mice using Pandas
clean_data.groupby('Sex').size().plot(kind="pie", figsize=(10,8), autopct="%.1f%%", colors=["b", "r"])
plt.ylabel("Number of mice")
plt.title("Total number of mice by gender")
plt.axis("Equal")
plt.show


# In[18]:


# Generate a pie plot showing the distribution of female versus male mice using pyplot
gender_data = clean_data.groupby('Sex').size()

plt.figure(figsize=(10,8))
plt.pie(gender_data, labels=gender_data.index, autopct="%.1f%%", colors=["b", "r"])
plt.ylabel("Number of mice")
plt.title("Total number of mice by gender")
plt.axis("Equal")
plt.show


# ## Quartiles, Outliers and Boxplots

# In[19]:


# Calculate the final tumor volume of each mouse across four of the treatment regimens:  
# Capomulin, Ramicane, Infubinol, and Ceftamin

# Start by getting the last (greatest) timepoint for each mouse
latest_timepoints = clean_data.groupby("Mouse ID")["Timepoint"].max()


# Merge this group df with the original DataFrame to get the tumor volume at the last timepoint
latest_mice = pd.merge(latest_timepoints, clean_data, on=['Mouse ID', 'Timepoint'])
latest_mice


# In[21]:


# Put treatments into a list for for loop (and later for plot labels)
treatments = ['Capomulin', 'Ramicane', 'Infubinol', 'Ceftamin']

# Create empty list to fill with tumor vol data (for plotting)
tumor_vol = {}

# Calculate the IQR and quantitatively determine if there are any potential outliers. 
for treatment in treatments:
    
    # Locate the rows which contain mice on each drug and get the tumor volumes
    data = latest_mice['Tumor Volume (mm3)'].loc[latest_mice['Drug Regimen'] == treatment]
    quartiles = data.quantile([0.25,0.5,0.75])
    lowerq = quartiles[0.25]
    upperq = quartiles[0.75]
    iqr = upperq - lowerq
    
    # add subset 
    tumor_vol[treatment] = data.values
    
    # Determine outliers using upper and lower bounds
    lower_bound = lowerq - (1.5 * iqr)
    upper_bound = upperq + (1.5 * iqr)
    lower_outliers = data.loc[data < lower_bound]
    upper_outliers = data.loc[data > upper_bound]
    
    print(f"In treatment {treatment}, there are {len(lower_outliers)} lower outliers and {len(upper_outliers)} upper outliers")


# In[23]:


# Generate a box plot that shows the distrubution of the tumor volume for each treatment group.
fig, ax = plt.subplots(figsize=(10,8))
outlierformat = {'marker': "d", 'markerfacecolor': "r", 'markersize': "15"}
ax.boxplot(tumor_vol.values(), flierprops=outlierformat)
ax.set_xticklabels(tumor_vol.keys())
plt.ylabel("Tumor Volume (mm3)")
plt.title("Final tumor volume by treatment")

plt.show


# ## Line and Scatter Plots

# In[43]:


# Generate a line plot of tumor volume vs. time point for a mouse treated with Capomulin
capomulin_data = clean_data.loc[clean_data['Drug Regimen'] == "Capomulin"]
mouse_id = capomulin_data['Mouse ID'].iloc[int(len(capomulin_data)/100*35)]

mouse_data = capomulin_data.loc[capomulin_data['Mouse ID'] == "l509"]

mouse_data.plot(x="Timepoint", y="Tumor Volume (mm3)", figsize=(10,8), marker="o")
plt.ylabel('Tumor Volume (mnm3)')
plt_label = "Tumor volume for mouse " + "l509" 
plt.title(plt_label)
xmax = mouse_data['Timepoint'].max() + 2
ymin = mouse_data['Tumor Volume (mm3)'].min() - 0.2
ymax = mouse_data['Tumor Volume (mm3)'].max() + 0.2
plt.xlim(-0.5,xmax)
plt.ylim(ymin,ymax)
plt.show


# In[26]:


# Generate a scatter plot of average tumor volume vs. mouse weight for the Capomulin regimen
plot_data = capomulin_data.groupby('Weight (g)')['Tumor Volume (mm3)'].mean()
plt.figure(figsize=(10,8))
plt.scatter(plot_data.index, plot_data.values)
plt.xlabel("Weight (g)")
plt.ylabel("Average tumor volume")
plt.title("Average tumor volume by mouse weight for the Capomulin regimen")
xmax = plot_data.index.max() + 2
xmin = plot_data.index.min() - 2
ymin = plot_data.values.min() - 0.5
ymax = plot_data.values.max() + 0.5
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.show


# ## Correlation and Regression

# In[32]:


# Calculate the correlation coefficient and linear regression model 
# for mouse weight and average tumor volume for the Capomulin regimen
slope, intercept, corrco, pv, se = st.linregress(plot_data.index, plot_data.values)
pearson, pearp = st.pearsonr(plot_data.index, plot_data.values)
print ("Pearson p value: " + str(pearp))
line_values = plot_data.index * slope + intercept


# In[41]:


plt.figure(figsize=(10,8))
plt.scatter(plot_data.index, plot_data.values)
plt.plot(plot_data.index, line_values)
plt.xlabel("Weight (g)")
plt.ylabel("Average tumor volume")
plt.title("Average tumor volume by mouse weight for the Capomulin regimen")
xmax = plot_data.index.max() + 2
xmin = plot_data.index.min() - 2
ymin = plot_data.values.min() - 0.5
ymax = plot_data.values.max() + 0.5
plt.show


# In[ ]:




