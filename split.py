import pandas as pd

# Reading input data
print('Reading input data...')
input_file = "data_preparation/users_full.csv"
input_data = pd.read_csv(input_file)

# Transforming boolean values into 0s and 1s
print('Transforming boolean values into 0s and 1s...')
input_data['BANKING'] = (input_data['BANKING']).astype(int)
input_data['DATA_USER'] = (input_data['DATA_USER']).astype(int)
input_data['TIKTOK'] = (input_data['TIKTOK']).astype(int)
input_data['FACEBOOK'] = (input_data['FACEBOOK']).astype(int)
input_data['FB_VIDEO'] = (input_data['FB_VIDEO']).astype(int)
input_data['WHATSAPP'] = (input_data['WHATSAPP']).astype(int)
input_data['INSTAGRAM'] = (input_data['INSTAGRAM']).astype(int)
input_data['SPOTIFY'] = (input_data['SPOTIFY']).astype(int)
input_data['AVG_NAV_AMNT_10'] = (input_data['AVG_NAV_AMNT_10']).astype(int)
input_data['AVG_NAV_AMNT_20'] = (input_data['AVG_NAV_AMNT_20']).astype(int)
input_data['AVG_NAV_AMNT_20PLUS'] = (input_data['AVG_NAV_AMNT_20PLUS']).astype(int)

# Removing columns that don't help
print('Removing columns that dont help...')
input_data = input_data.drop('MSISDN', axis=1)
input_data = input_data.drop('GENDER', axis=1)
input_data = input_data.drop('FAV_HOUR', axis=1)
input_data = input_data.drop('LAST_DATA_USE', axis=1)

print('Splitting data intro train, test, validation sets...')
data_training = input_data.sample(frac=0.7)
data_test = input_data.drop(data_training.index)
data_training2 = data_training.sample(frac=0.7)
data_validation = data_training.drop(data_training2.index)

data_size = len(input_data["MSG_CLICKED"])
data_clicked = input_data["MSG_CLICKED"].sum()
data_clicked_percentage = data_clicked * 100 / data_size
print("Data Size: " + str(data_size))
print("Data Clicked: " + str(data_clicked))
print("Data %: " + str(data_clicked_percentage))

training_size = len(data_training2["MSG_CLICKED"])
training_clicked = data_training2["MSG_CLICKED"].sum()
training_clicked_percentage = training_clicked * 100 / training_size
print("Training Size: " + str(training_size))
print("Amount Clicked: " + str(training_clicked))
print("Clicked %: " + str(training_clicked_percentage))

test_size = len(data_test["MSG_CLICKED"])
test_clicked = data_test["MSG_CLICKED"].sum()
test_clicked_percentage = test_clicked * 100 / test_size
print("Test Size: " + str(test_size))
print("Amount Clicked: " + str(test_clicked))
print("Clicked %: " + str(test_clicked_percentage))

validation_size = len(data_validation["MSG_CLICKED"])
validation_clicked = data_validation["MSG_CLICKED"].sum()
validation_clicked_percentage = validation_clicked * 100 / validation_size
print("Validation Size: " + str(validation_size))
print("Amount Clicked: " + str(validation_clicked))
print("Clicked %: " + str(validation_clicked_percentage))

# CODE USED TO GENERATE TRAINING AND TEST FILES
print('Exporting sets into .csv files...')
data_training2.to_csv('features_2/training.csv', index=False)
data_validation.to_csv('features_2/validation.csv', index=False)
data_test.to_csv('features_2/test.csv', index=False)
