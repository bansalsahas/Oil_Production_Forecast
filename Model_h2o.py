import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sn
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel("oil_balance.xlsx")

data.describe()

data.info()

data.head(5)

"""Deleting columns with nulls"""

data= data.drop(columns=['GDEBTSA','PPISA', 'PRCUS', 'PUCUS'])

"""Adding weekly, biweekly and monthly avg oil production to the dataset."""

data['oilprod_week_avg'] = data['INDPROD'].rolling(7).mean()
data['oilprod_2weeks_avg'] = data['INDPROD'].rolling(14).mean()
data['oilprod_month_avg'] = data['INDPROD'].rolling(30).mean()

#Extracting year,month and date from the Date field.
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day

data.rename(columns = {'INDPROD':'int_prod','REFINOBS':'ref_intake','GDEBTSA':'govt_debt','EXPSA':'exp_sa','IPSA':'imp_sa','PRCUS':'pvtcons_us',
                        'PUCUS':"pubcons_us",'total_prod':'total_export','REFINOBS':'ref_intake','OILPRODSA':'saudi_oil_prod','OILPRODUS':'usa_oil_prod'},inplace = True)

data.sum().isnull()

data = data[data.Date.notnull()]

data

"""No null values are present in the dataset

**EDA**
"""

data[['Date','saudi_oil_prod','total_export']].set_index('Date').plot(linewidth=1.0)
plt.title("Oil Production vs. Total Export Trend")

"""There is a significant decline in oil production and exports during February and March of 2020 due to Covid-19."""

data[['Date','saudi_oil_prod','usa_oil_prod']].set_index('Date').plot(linewidth=1.0)
plt.title("Oil Production Trends of Saudi Arabia & US")

"""Comparing Oil Production trends of US and Saudi Arabia, it is evident that Saudi Arabia has been consistent with oil production throughout the years. US started catching up with Saudi's oil production after 2015. After mid 2018, US surpassed the oil production of Saudi Arabia and has been doing better even after the pandemic. Also, we observe that when US production increases there is a decrease in Saudi Arabia Oil production.

**Creating train test split**
"""

data.reset_index(drop=True,inplace=True)
train = data.loc[:int(data.shape[0]*0.8),:]
test = data.loc[int(data.shape[0]*0.8):,:]

train

"""Plotting train and test split"""

plt.plot(train.index,train['saudi_oil_prod'])
plt.plot(test.index,test['saudi_oil_prod'])
plt.ylabel('saudi_oil_prod',fontsize=18)
plt.legend(['train','test'])
plt.show()

import h2o
from h2o.automl import H2OAutoML

h2o.init(nthreads=-1)

#!npm install -g localtunnel -qq > /dev/null

#get_ipython().system_raw('lt --port 54321 >> url.txt 2>&1 &')

#!cat url.txt

"""**Converting pandas dataframe to h2o dataframe**"""

hf_train = h2o.H2OFrame(train)
hf_test = h2o.H2OFrame(test)

hf_train.describe()

y = 'saudi_oil_prod'
X = hf_train.columns
X.remove(y)

X

aml = H2OAutoML(max_runtime_secs = 600,
                seed = 42)
aml.train(x = X, 
          y = y,
          training_frame = hf_train,
          leaderboard_frame = hf_test)

leader_model = aml.leader

hf_test_predict = leader_model.predict(hf_test)

hf_test_predict.head(10)

df_results = pd.DataFrame()
df_results['ground_truth'] = test['saudi_oil_prod'].reset_index(drop=True)
df_results['predictions'] = h2o.as_list(hf_test_predict,use_pandas=True)
df_results.head()

from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(x=df_results['predictions'],y=df_results['ground_truth'])
print('R2 = ',r_value*r_value)

def wmape_gr(df_results, ground_truth, predictions):
    # we take two series and calculate an output a wmape from it
    # make a series called mape
    se_mape = abs(df_results[ground_truth] - df_results[predictions]) / df_results[ground_truth]
    # get a float of the sum of the actual
    ft_actual_sum = df_results[ground_truth].sum()
    # get a series of the multiple of the actual & the mape
    se_actual_prod_mape = df_results[ground_truth] * se_mape
    # summate the prod of the actual and the mape
    ft_actual_prod_mape_sum = se_actual_prod_mape.sum()
    # float: wmape of forecast
    ft_wmape_forecast = ft_actual_prod_mape_sum / ft_actual_sum
    # return a float

    return ft_wmape_forecast
  
wmape_gr(df_results,'ground_truth','predictions')
  
plt.scatter(x=df_results['predictions'],y=df_results['ground_truth'],s=1)
plt.xlabel('predictions',fontsize=18)
plt.ylabel('ground_truth',fontsize=18)
plt.show()

plt.plot(df_results['ground_truth'])
plt.plot(df_results['predictions'])
plt.ylabel('Saudi Oil Production',fontsize=18)
plt.legend(['ground_truth','prediction'])
plt.show()

model_path = h2o.save_model(model=leader_model, path="/content/mymodel", force=True)
saved_model = h2o.load_model(model_path)

h2o.cluster().shutdown()
