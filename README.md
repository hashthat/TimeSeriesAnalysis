# TimeSeriesAnalysis

## INTRODUCTION
 In search of a dataset I have always loved coffee. I used to have a dream of running a coffee shop or a roastery, but never have done so. Searching through Kaggle’s datasets I found some simulated data for a fictitious business Marvin Roasters. I chose this over other datasets because of the nature of the assignment being Time Series Analysis where creating a machine learning model to predict the daily sales of the business was worth the invested learning. So, that is what I went with.

Something someone could do is expand on the dataset for purposes of predicted seasonal sales. Given they would be a good prompt engineer someone could definitely expand on this dataset for their own practice or purpose. I actually didn’t realize the dataset was only for six months of data, but for this project and the practice of Machine Learning it was enough given my background in the restaurant industry the dataset made sense for my own personal predictions from practical real world experience as the hindsight to the understood task to creating an observation of day to day business trends for a business in the service industry. Overall I found the data to be accurate intuitively by design, and I’m glad I had the opportunity to learn. 
## DATASET
The dataset I chose was based on simulated Coffee Shop sales from “Mavin Roasters” over a time period of six months. I found the dataset on Kaggle (https://www.kaggle.com/code/ahmedabbas757/coffee-shop-sales/input) where I had to format the data using the Pandas and Numpy libraries. I had to do some thinking and combined some of the data:
"""Convert 'transaction_date' to datetime and check order"""
df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%m/%d/%Y')
df['transaction_cost'] = df['transaction_qty'] * df['unit_price']
print(df[['transaction_date', 'transaction_time', 'transaction_cost']].head(1000))

With this the data was transcribed to be understood as the transaction date would be seen along with the transaction cost which was formatted to be the number of items sold multiplied by the unit price. This helped the data be presented in a way I could accomplish the goal of understanding the overall sales in a single day. In the end, I used the data to view the daily threshold to help predict the daily sales or weekly revenue for the coffee roasting company.
## SELECTED MODEL
 I chose the LSTM (Long-Short-Term-Memory) model for my experiment. I Found this best given the data will reflect business rushes and seasonal changes in sales which would help with the prediction. The model can also evaluate daily and weekly sales using a sliding window scale, which means the data of different seasons or months can be viewed to help with the models accuracy and predictions.
## ANALYSIS RESULTS
The daily analysis of sales were interesting because there was the graph I did use showed there was an increase in sales between 8am-10am where the threshold or median of the sales occurred between 2pm and 4pm (happy hour). Overall the predictions were accurate but there was a pattern I noticed as the training and testing occurred. The accuracy was great the first time through but there’s a problem I ran into as far as accuracy goes. Because the simulated data is not based on annual sales with two years of data it is hard to predict what type of seasons bring in more sales. I would say from being someone who loves coffee sales would increase naturally during the holidays, or in the fall, and the data that we have I know restaurants and retail business stores usually come to a halt in January and February. The seasons bring in different types of events as people are doing different things as far as an annual observation goes. But, this data set is based on six months of sales, which means there really isn’t enough data to bring any intuitive thought or implementation to the data. So the problem with the test results vs the training results had to do with the model “assuming” sales because there wasn’t enough data for it to know more than what was given. The y_train and y_test had to do with whether or not the daily sales would meet the threshold in my model. When the prediction was true multiple times in a row the prediction continued to test true even when the true data showed the daily sales weren’t meeting the daily threshold as far as the true data goes. So, my model had a lot of false predictions. The way I helped the training was with the train_test_split through sklearn for the model selection. This helped randomize and shuffle the data so the prediction was less predictable from the data the model was already trained on. This helped the overall accuracy eliminating false positives in the testing phase of the evaluation.
from sklearn.model_selection import train_test_split


"""Shuffle & split in one step"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,   # 20% test
    shuffle=True,    # Shuffle before splitting
    random_state=42 # For reproducibility
)


## CONCLUSION
 Despite the odds, I found the model to predict some interesting insight. It proves when data is formatted in a specific way where it can be viewed with a question such as “what’s the threshold of sales” for a given season, or on the daily by the hour. What’s the pace, whens the rush hour. There are answers to the sales and revenue of the business that can be seen in the numbers. Another question might be what is purchased more frequently and in what season. These types of answers in data can help someone predict the outcome of increasing revenue or creating new opportunities through their chosen business model. If someone wanted to gain more insight they could add customer reviews and still based on that data find a question, train a model that would reflect a suggestion for what a company might do differently still… with the sales in mind.
	Even with the restriction of having only six months,  I think the behaviour of the model was great. It was a learning experience and it was interesting to learn about the false positives because of the limited data. Overall the model trained and tested well offering a loss of .36 which isn’t bad in the grand scheme of things.
## REFERENCES
Abbas, A. (2023). Coffee Shop Sales [Data set and notebook]. Kaggle. Retrieved June 22, 2025, from https://www.kaggle.com/code/ahmedabbas757/coffee-shop-sales

Pierian Training. (2022). TensorFlow LSTM example: A beginner’s guide. Retrieved June 22, 2025, from https://pieriantraining.com/tensorflow-lstm-example-a-beginners-guide/

GeeksforGeeks. (2025, May 28). Deep learning: Introduction to Long Short‑Term Memory. Retrieved June 22, 2025, from https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/
