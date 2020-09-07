# RobinHood-Code
How to connect to RobinHood and stock collect data every 3 minutes for both your watch list and Holding Stocks. 


This Project is a Work in Progress. The current code is a working model.
Some of the functions available:
1)  connect automatically to the RobinHood API's using Credentials and OTP
2)  get Infromation about the Holding and WatchList stocks in your app.
3) collate data for both and store it in a local postgres database.
4) The code also connects with Twilio and sends Messages when the daily low or the yearly low (or is very close) is reached for a stock.Similarly for the yearly High/daily high
5) The Code calulates Simple Moving average based on last 20 day records. Also, calculates Bolinger Bands for each stock. Finally, it provides Momentum of last 5 trades.

As mentioned this code will be refined and more functionality will be added.
