import robin_stocks as r 
import numpy as np
import pyotp
from twilio.rest import Client
from pandas.io.json import json_normalize
import pandas as pd
from datetime import datetime
import pyodbc,psycopg2 
import sys
import glob, os, os.path

begin_time = datetime.now()
#Authenticate(MFA) the Robinhood account using pyotp 
totp  = pyotp.TOTP("YOur MFA Code").now()
login = r.authentication.login('youremailID@email.com','yourPAssword',mfa_code=totp)	

#Connect with the posgres SQL client using psycopg2
conn = None
try:
    conn = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="postgres")
    cur = conn.cursor()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)


def addSymbol(positions_data):
    for index,item in positions_data.iterrows():
        positions_data.loc[index,'symbol'] = r.get_symbol_by_url(item['instrument'])
    return positions_data

def get_latest_price(positions_data, includeExtendedHours=True):
    for index,item in positions_data.iterrows():
        positions_data.loc[index,'Current_Price'] = r.get_latest_price(item['symbol'])[0]
    return positions_data

#get fundamental information about the stock 
def fundamental_information(positions_data):
    fund_stock = []
    stock =[]
    for index,item in positions_data.iterrows():
        fund_stock =(json_normalize(r.stocks.get_fundamentals(item['symbol']))) 
        date = datetime.now()
        for k,v in fund_stock.iterrows():
            positions_data.loc[index,'average_volume'] = v['average_volume']
            positions_data.loc[index,'ceo'] = v['ceo']
            positions_data.loc[index,'datetime'] = date
            positions_data.loc[index,'dividend_yield'] = v['dividend_yield']
            positions_data.loc[index,'float'] = v['float']
            positions_data.loc[index,'headquarters_city'] = v['headquarters_city']
            positions_data.loc[index,'headquarters_state'] = v['headquarters_state']
            positions_data.loc[index,'high'] = v['high']
            positions_data.loc[index,'high_52_weeks'] = v['high_52_weeks']
            positions_data.loc[index,'industry'] = v['industry']
            positions_data.loc[index,'low'] = v['low']
            positions_data.loc[index,'low_52_weeks'] = v['low_52_weeks']
            positions_data.loc[index,'market_cap'] = v['market_cap']
            positions_data.loc[index,'num_employees'] = v['num_employees']
            positions_data.loc[index,'open'] = v['open']
            positions_data.loc[index,'pb_ratio'] = v['pb_ratio']
            positions_data.loc[index,'pe_ratio'] = v['pe_ratio']
            positions_data.loc[index,'sector'] = v['sector']
            positions_data.loc[index,'shares_outstanding'] = v['shares_outstanding']
            positions_data.loc[index,'symbol'] = v['symbol']
            positions_data.loc[index,'volume'] = v['volume']
            positions_data.loc[index,'year_founded'] = v['year_founded']

    return positions_data

def get_position_data():
    positions_data = r.get_open_stock_positions()
    positionData = pd.DataFrame(positions_data)
    condensedpositionData = positionData[['average_buy_price','quantity','instrument']]
    condensedpositionData['symbol'] = ''
    condensedpositionData['Current_Price'] = ''
    condensedpositionData['Holding'] = 'Yes'
    addSymbol(condensedpositionData)
    get_latest_price(condensedpositionData)
    return fundamental_information(condensedpositionData)



def get_watchlist_data():
    watchListData = r.account.get_watchlist_by_name(name = 'My First List')
    WatchListSymbols = json_normalize(watchListData['results'])
    symbol_series = WatchListSymbols['symbol'].to_frame()
    symbol_series['Current_Price'] = ''
    symbol_series['Holding'] = 'No'
    watchList_dataframe = get_latest_price(symbol_series)
    Latest_WatchList_Information_Dataset = fundamental_information(watchList_dataframe)
    return fundamental_information(Latest_WatchList_Information_Dataset)

#calculate SMA and Bollinger Bands
#get Values for past 20 days
def Metrics(Combined_data):
    query_last_20_days = """SELECT * FROM 
    public.historystockdata hist
    JOIN
    public.time time on (hist.datetime = time.created_datetime) 
    WHERE hist.datetime> current_date - interval '20' day
	AND cast (hist.datetime::timestamp as time) Between '04:00:00' AND '20:00:00'"""

    cur.execute(query_last_20_days)
    data = cur.fetchall()

    num_fields = len(cur.description)
    field_names = [i[0] for i in cur.description]
    #print(field_names)

    last20days = pd.DataFrame(data,columns = field_names )
    if len(last20days.index)!= 0: 
		#print(last20days)
        last20days.columns
        data_price = last20days[['symbol','current_price']]
        data_volume = last20days[['symbol','volume']]
        Combined_data_SMA_price = data_price.groupby(['symbol']).mean()
        Combined_data_SMA_price.reset_index(inplace = True)
        #Combined_data['SMA_volume'] = data_volume.groupby(['symbol']).mean()
        merged_inner = pd.merge(left=Combined_data, right=Combined_data_SMA_price, left_on='symbol', right_on='symbol')
		
        #print(SMA_price)
        Combined_data_standard_dev_price = data_price.groupby(['symbol']).std()
        Combined_data_standard_dev_price.reset_index(inplace = True)
        #Combined_data['standard_dev_volume'] = data_volume.groupby(['symbol']).std()
        merged_inner_final = pd.merge(left=merged_inner, right=Combined_data_standard_dev_price, left_on='symbol', right_on='symbol')
		
		#print(merged_inner_final)
		#bollingerBand = Combined_data[['symbol','Current_Price']] - SMA /standard_dev
		#print(bollingerBand)
		#merged_inner_final['bb'] = (merged_inner_final.Current_Price - merged_inner_final.current_price_x )/(2*merged_inner_final.current_price_y)
        return merged_inner_final
    else:
        Combined_data['current_price_x'] = 0.0
        Combined_data['current_price_y'] = 0.0
        return Combined_data



#get Data from both WatchList and Latest Stock Data
def Combine_data():
    WatchList_data = get_watchlist_data()
    Position_data = get_position_data()
    Combined_data = pd.concat([Position_data, WatchList_data])
    #Metrics(Combined_data)
    #print(t)
    #Combined_data['SMA'] = SMA
    #Combined_data['standard_dev'] =standard_dev
    #Combined_data['bollingerBand'] = bollingerBand
    return Metrics(Combined_data)



#transfer Data to CSV.
def LatestDataToCSV(StockData):
    directory='C:/Stock_data/Latest_stock_data/*'
    filelist = glob.glob(directory)
    for f in filelist:
        os.remove(f)
    date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
    name = 'C:/Stock_data/Latest_stock_data/stock_data_Latest.csv'
    StockData.to_csv(name)

#LatestDataToCSV(Combine_data())

def HistoricalDataToCSV(StockData):
    LatestDataToCSV(StockData)
    root_dir = 'C:/Stock_data/Historical_stock_data/'
    today = datetime.now()
    year = today.strftime("%Y")
    month=today.strftime("%m")
    day=today.strftime("%d")
    output_dir = root_dir  + year +"/" + month + "/" + day
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
    filename = output_dir+'/stock_data %s.csv'%date
    StockData.to_csv(filename)
combData = Combine_data()
HistoricalDataToCSV(combData)


conn.commit()
#conn = pyodbc.connect('Driver={SQL Server};'
#                      'Server=5CD5133134\MSSQLSERVER2017;'
#                      'Trusted_Connection=yes;')
#conn.autocommit = True
#cur = conn.cursor()


LatestStockDataTable_Drop = "DROP TABLE if exists LastestStockData"

LatestStockDataTable = ("""CREATE TABLE if not exists LastestStockData(
CreatedDate timestamp,
Holding text,
average_buy_price float,
quantity float,
symbol text UNIQUE,
Current_Price float,
ceo text,
dividend_yield float,
headquarters_city text,
headquarters_state text,
stock_high float,
stock_low float,
low_52_weeks float,
high_52_weeks float,
industry text,
market_cap text,
sector text,
volume float,
year_founded int,
SMA float,
std_dev float
)
""")


LatestStockDataTable_Insert = """ insert into LastestStockData (
 CreatedDate
,Holding 
,average_buy_price 
,quantity 
,symbol
,Current_Price 
,ceo 
,dividend_yield  
,headquarters_city 
,headquarters_state 
,stock_high 
,stock_low 
,low_52_weeks 
,high_52_weeks
,industry 
,market_cap 
,sector 
,volume
,year_founded
,SMA
,std_dev) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT(symbol) DO NOTHING"""

combData = Combine_data()

def Send_Alert(result):
	result['close_to_daily_low'] = np.where((result['Current_Price'].astype(float) <= result['low'].astype(float)*1.0005), 'True','False')
	result['close_to_year_low'] = np.where((result['Current_Price'].astype(float) <= result['low_52_weeks'].astype(float)*1.01), 'True','False')
	result['close_to_daily_high'] = np.where((result['Current_Price'].astype(float) >= result['high'].astype(float)*1.0005), 'True','False')
	result['close_to_year_high'] = np.where((result['Current_Price'].astype(float) >= result['high_52_weeks'].astype(float)*1.01), 'True','False')

	s = result[['symbol','close_to_daily_low','close_to_year_low','low','low_52_weeks','Current_Price','close_to_daily_high','close_to_year_high']]
	t1 = s.loc[(s['close_to_daily_low'] == 'True')] 
	t2 = s.loc[(s['close_to_year_low'] == 'True')]
	t3 = s.loc[(s['close_to_daily_high'] == 'True')] 
	t4 = s.loc[(s['close_to_year_high'] == 'True')]
	
	t11 = t1[['symbol','Current_Price']]
	t12 = t2[['symbol','Current_Price']]
	
	t13 = t3[['symbol','Current_Price']]
	t14 = t4[['symbol','Current_Price']]
	
	
	#t2['symbol','price']
	#current price is equal to low +- 1%
	
	#	int(WatchListSymbols['low'])*1.1
	# the following line needs your Twilio Account SID and Auth Token
	client = Client("clientInformation", "Passcode")

	# change the "from_" number to your Twilio number and the "to" number
	# to the phone number you signed up for Twilio with, or upgrade your
	# account to send SMS to any phone number
	
	'''if len(t11.index) != 0: 
		client.messages.create(to="+15202756912",from_="+13132468061",body="Daily low{}".format(t11.to_string()))
	if len(t12.index) != 0: 
		client.messages.create(to="+15202756912",from_="+13132468061",body="Yearly Low{}".format(t12.to_string()))
	if len(t13.index) != 0: 
		client.messages.create(to="+15202756912",from_="+13132468061",body="Daily High{}".format(t13.to_string()))
	if len(t14.index) != 0: 
		client.messages.create(to="+15202756912",from_="+13132468061",body="Yearly High{}".format(t14.to_string()))
    '''
Send_Alert(combData)	
print(combData.columns)
cur.execute(LatestStockDataTable_Drop)
cur.execute(LatestStockDataTable)


conn.commit()
for i, row in combData.iterrows():
    row = row.fillna('0')
    cur.execute(LatestStockDataTable_Insert,(
     row.datetime
    ,row.Holding
    ,row.average_buy_price
    ,row.quantity
    ,row.symbol
    ,row.Current_Price
    ,row.ceo
    ,row.dividend_yield
    ,row.headquarters_city
    ,row.headquarters_state
    ,row.high
    ,row.low
    ,row.low_52_weeks
    ,row.high_52_weeks
    ,row.industry
    ,row.market_cap
    ,row.sector
    ,row.volume
    ,row.year_founded
    ,row.current_price_x
    ,row.current_price_y

))

conn.commit()


# In[285]:

conn.commit()
time_table_create = ("""
CREATE TABLE if not exists time (
 created_dateTime timestamp NOT NULL PRIMARY KEY
, hour int NOT NULL
, day int NOT NULL
, week int NOT NULL
, month int NOT NULL
, year int NOT NULL
, weekday int NOT NULL);
""")


#LatestStockDataTable_Drop = "DROP TABLE if exists time"
#cur.execute(LatestStockDataTable_Drop)

time_table_insert = ("""
insert into time (created_dateTime, hour, day, week, month, year, weekday) 
values(%s, %s, %s, %s, %s, %s, %s)   on conflict(created_dateTime) do nothing  
""");

t = combData.copy()
t['ts'] = pd.to_datetime(t['datetime'], unit='ms')    
    
# insert time data records
time_data = [t.ts, t.ts.dt.hour, t.ts.dt.day, t.ts.dt.week, t.ts.dt.month, t.ts.dt.year, t.ts.dt.weekday]
column_labels = ['start_time', 'hour', 'day', 'week', 'month', 'year', 'weekday']
time_df = pd.DataFrame.from_dict(dict(zip(column_labels, time_data)))
#time_df
cur.execute(time_table_create)
time_df.dtypes

for i, row in time_df.iterrows():
   # print(row)
    cur.execute(time_table_insert, (row.start_time,row.hour,row.day,row.week,row.month,row.year,row.weekday))



conn.commit()
CompanyData = """create table if not exists company_stock_Information
(Holding text
,average_buy_price float 
,quantity float
,symbol varchar(100) not null Primary Key
,ceo text 
,dividend_yield  float
,headquarters_city text
,headquarters_state text
,industry text
,market_cap float
,sector text
,volume float
,year_founded int)"""

CompanyDropTable = "Drop Table if exists company_stock_Information"

#ur.execute(CompanyDropTable)
cur.execute(CompanyData)
company_stock_insert = """ insert into company_stock_Information(
Holding 
,average_buy_price 
,quantity 
,symbol
,ceo 
,dividend_yield  
,headquarters_city 
,headquarters_state 
,industry 
,market_cap 
,sector 
,volume
,year_founded) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)   on conflict(symbol) 
DO UPDATE SET Holding = EXCLUDED.Holding, average_buy_price = EXCLUDED.average_buy_price, quantity = EXCLUDED.quantity, dividend_yield = EXCLUDED.dividend_yield,
volume = EXCLUDED.volume;  
"""

for i, row in combData.iterrows():
    row = row.fillna('0')
    cur.execute(company_stock_insert,(
    row.Holding
    ,row.average_buy_price
    ,row.quantity
    ,row.symbol
    ,row.ceo
    ,row.dividend_yield
    ,row.headquarters_city
    ,row.headquarters_state
    ,row.industry
    ,row.market_cap
    ,row.sector
    ,row.volume
    ,row.year_founded))
conn.commit()



conn.commit()
HistoryStockData = """create table if not exists HistoryStockData
(stockprice_id serial Unique
,datetime timestamp references time(created_dateTime)
,current_price float
,open_stock float
,symbol varchar(100) references company_stock_Information(symbol)
,volume float
,primary key(datetime,symbol)
)"""

HistoryDropTable = "Drop Table if exists HistoryStockData"

#cur.execute(HistoryDropTable)
cur.execute(HistoryStockData)
HistoryStockInsert = """ insert into HistoryStockData( 
datetime 
,current_price 
,symbol
,volume
,open_stock
) values(%s,%s,%s,%s,%s) ON CONFLICT(symbol,datetime) DO UPDATE SET current_price = EXCLUDED.current_price,volume = EXCLUDED.volume"""

for i, row in combData.iterrows():
    row = row.fillna('0')
    cur.execute(HistoryStockInsert,(
     pd.to_datetime(row['datetime'], unit='ms')
    ,row.Current_Price
    ,row.symbol
    ,row.volume
	,row.open
    ))
conn.commit()


def five_momentum():
    momentum_five_query = """SELECT r.symbol symbol,(r.current_price/r1.open_stock) momentum
from
(SELECT rs.symbol,rs.datetime,rs.current_price,rs.open_stock
    FROM (
        SELECT symbol, datetime,current_price,open_stock, Rank() 
          over (Partition BY Symbol
                ORDER BY datetime DESC ) AS Rank
        FROM public.historystockdata
        ) rs WHERE Rank <= 5) r
JOIN

(SELECT rs.symbol symbol, rs.open_stock open_stock
    FROM (
        SELECT symbol, datetime,current_price,open_stock, Rank() 
          over (Partition BY Symbol
                ORDER BY datetime DESC ) AS Rank
        FROM public.historystockdata
        ) rs WHERE Rank <= 1) r1
	ON 
	r1.symbol = r.symbol;"""

    cur.execute(momentum_five_query)
    data = cur.fetchall()

    num_fields = len(cur.description)
    field_names = [i[0] for i in cur.description]
    #print(field_names)

    five_momentum = pd.DataFrame(data,columns = field_names)
    five_momentum1 =five_momentum['momentum']
    five_momentum2 =five_momentum['symbol']
    dups = five_momentum2.drop_duplicates( keep='first', inplace=False)
    #print(dups)
    return (pd.DataFrame(five_momentum1.values.reshape(-1, 5), index = dups,
                   columns=['momentum1','momentum2','momentum3','momentum4','momentum5']))
momentum = five_momentum()  
time = datetime.now()
momentum_five_insert = ("""
insert into momentum_five(symbol, datetime,momentum_1,momentum_2,momentum_3,momentum_4,momentum_5) 
values(%s, %s, %s, %s, %s, %s, %s)   
""");

for i, row in momentum.iterrows():
    cur.execute(momentum_five_insert, (i,time,row.momentum1,row.momentum2,row.momentum3,row.momentum4,row.momentum5))

drop_momentum_five = ("""Drop table if exists momentum_five_latest""")
create_momentum_five_latest = ("""create table if not exists momentum_five_latest(symbol varchar(100),datetime timestamp, momentum_1 float, momentum_2 float, momentum_3 float, momentum_4 float, momentum_5 float);
""")

cur.execute(drop_momentum_five)

cur.execute(create_momentum_five_latest)

momentum_five_insert = ("""
insert into momentum_five_latest(symbol, datetime,momentum_1,momentum_2,momentum_3,momentum_4,momentum_5) 
values(%s, %s, %s, %s, %s, %s, %s)   
""");

for i, row in momentum.iterrows():
    cur.execute(momentum_five_insert, (i,time,row.momentum1,row.momentum2,row.momentum3,row.momentum4,row.momentum5))

	
conn.commit()

cur.close()

print(datetime.now() - begin_time)










