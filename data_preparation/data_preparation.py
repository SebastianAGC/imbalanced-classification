import pandas as pd
import numpy as np

users = pd.read_csv(r'USUARIOS.csv')
print (users)

users_first_msg = pd.read_csv('users_first_msg.csv', header=None)
print(users_first_msg)

users_first_msg_clicked = pd.read_csv('users_first_msg_clicked.csv', header=None)
print(users_first_msg_clicked)

users_second_msg = pd.read_csv('users_second_msg.csv', header=None)
print(users_second_msg)

users_second_msg_clicked = pd.read_csv('users_second_msg_clicked.csv', header=None)
print(users_second_msg_clicked)

#Adding columns to users dataframe
users["FIRST_MSG_SENT"] = np.nan
users["FIRST_MSG_CLICKED"] = np.nan
users["SECOND_MSG_SENT"] = np.nan
users["SECOND_MSG_CLICKED"] = np.nan
users["MSG_CLICKED"] = np.nan
print(users)
#         input_data.at[i, 'USER_AGE'] = age
for i in range(len(users)):
    print(i)
    #Checking if user was sent first message type
    if users["MSISDN"][i] in users_first_msg[0].values:
        users.at[i, "FIRST_MSG_SENT"] = 1
    else:
        users.at[i, "FIRST_MSG_SENT"] = 0
    
    #Checking if user clicked first message type
    if users["MSISDN"][i] in users_first_msg_clicked[0].values:
        users.at[i, "FIRST_MSG_CLICKED"] = 1
    else:
        users.at[i, "FIRST_MSG_CLICKED"] = 0
    
    #Checking if user was sent second message type
    if users["MSISDN"][i] in users_second_msg[0].values:
        users.at[i, "SECOND_MSG_SENT"] = 1
    else:
        users.at[i, "SECOND_MSG_SENT"] = 0
    
    #Checking if user clicked second message type
    if users["MSISDN"][i] in users_second_msg_clicked[0].values:
        users.at[i, "SECOND_MSG_CLICKED"] = 1
    else:
        users.at[i, "SECOND_MSG_CLICKED"] = 0
        
    if users["FIRST_MSG_CLICKED"][i] == True or users["SECOND_MSG_CLICKED"][i] == True:
        users.at[i, "MSG_CLICKED"] = 1
    else:
        users.at[i, "MSG_CLICKED"] = 0
        
print(users)

users.to_csv('users_full.csv')