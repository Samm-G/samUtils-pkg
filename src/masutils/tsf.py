class TSF:
    def __init__(self):
        """
        Theory:
            
            Intervals of Time Series:
                ● Yearly: GDP, Macro-economic series
                ● Quarterly: Revenue
                ● Monthly: Sales, Expenditure
                ● Weekly: Demand
                ● Daily: Closing price of stock
                ● Hourly: AAQI
                
            Special Features of Time Series:
                ● Data are not independent
                ● One defining characteristic of time series is that this is a list of observations where the ordering matters.
                ● Ordering is very important because there is dependency and changing the order will change the data structure
                ● Time Series does not admit missing data
                ● All data observations must be contiguous
                ● Impute missing data to the best of your knowledge
            
            Time Series Components:
                ● Systematic Components:
                    ● Trend Component
                    ● Seasonal Component
                ● Irregular Component (Noise)
                
            Decomposing a Time Series:
                ● To understand revenue generation without the quarterly effects
                    ● De-seasonalize the series
                    ● Estimate and adjust by seasonality
                ● Compare the long-term movement of the series (Trend) vis-a-vis short-term movement (seasonality) to understand which has the higher influence
                
                ● Additive model: 		
                    Observation = Trend + Seasonality + Error
                    ● Examples:
                        Average Monthly Temp: Constant seasonality, Trend contribution minimal
                        Quarterly Turnover
                        Champagne Sales
                        
                ● Multiplicative model: 	
                    Observation = Trend * Seasonality * Error
                    
                    ● Need logarithmic transformation to convert into an additive series:
                        Log(Vol) = log(Trend) + log(Seasonality) + log(Error)
                        
                    ● Examples:
                        Passenger Volume
                        

            Very Long Rage Forecasts Dont Work Well:
                
                ● Forecasts are done under the assumption that the market and other conditions in future are very much like the present
                ● Not that there will be no change in the market
                ● But the change is gradual, not a drastic change
                ● A financial crash like 2008 US market will send all forecasts into a tizzy
                ● Events like Demonetization would throw the forecasts into disarray
                
            Types fo Forecasts:
                
                Simple Forecast:
                    ● Naïve Forecast: Use the last observed value
                    ● Average Forecast: 
                    ● Moving Average Forecast:
                        ● Take average over a window of certain width
                        ● Move the window
                    ● None of these work well even for the most regular series

            Measures of Forecast Accuracy: (Performance Metrics):
                ● Residual sum of squares 
                    RSS = σ(Y(t) − Y^(t))2
                ● Mean sum of squares 
                    MSS = 1/T * σ(Y(t) − Y^(t))2
                ● Mean absolute deviation
                    MAD =1/T * (σ|Y(t) − Y^(t)|2)
                ● Mean absolute percent error
                    MAPE = 1/T σ[ |Y(t)− Y^(t)| / Y(t) ] x 100
                ● AIC:
                    AIC = 2k - 2ln(L^)
                        - k = no of estimated parameters in model
                        - L^ = Max value of likelihood function.
                        
            Exponential Smoothing Models:
            
                ● Simple Exponential Smoothing (SES):
                    F(t+1) = yt + α(-yt + Ft)
                        - α = Smoothing Constant
                        
                ● Double Exponential Smoothing (Holt Model):
                    L(t+1) = αFt + (1-α)yt
                    T(t+1) = β(L(t+1) - L(t)) + (1-β)Tt
                    
                    ● Applicable when data has Trend but no seasonality
                    ● An extension of SES
                    ● Two separate components are considered: Level and Trend
                    ● Level is the local mean
                    ● One smoothing parameter α corresponds to the level series
                    ● A second smoothing parameter β corresponds to the trend series
                    ● Also known as Holt model

                ● Triple Exponential Smoothing: (Exponential Smoothing with Seasonality) (Holt-Winters Model):
                    S(t+1) = γ (Y(t+1)/L(t+1)) + (1-γ)S(t+1-m)
                    L(t+1) = α(y(t+1) - S(t+1-m)) + (1-α)(L(t+1) + Tt )
                    T(t+1) = 1/M * Sigma(Y(m+1) - Y(t) / M)
            
            ● Stationary Series Features:
                ● A stationary time series is one whose properties do not depend on the time at which the series is observed. 
                ● The series with Trend or Seasonality are not sytationary.
                ● Mean of the time series will be a constant.
                ● Time series with trends are not stationary
                ● Variance of the time series will be a constant
                ● The correlation between the t-th term in the series and the t+m-th term in the series is constant for all time periods and for all m
                ● Another name: White Noise
            
            Steps for Analysis of Time Series:
            
                ● Visualization:
                ● Stationarization:
                    ● Do a formal test of hypothesis (ADF test)
                    ● If series non-stationary, stationarize
                ● Explore Autocorrelations and Partial Autocorrelations:
                ● Build ARIMA Model:
                    ● Identify training and test periods
                    ● Decide on model parameters
                    ● Compare models using accuracy measures
                    ● Make prediction
                    
            Difference Series:
                ● It is possible to make a non-stationary series stationary by taking differences between consecutive observations.
                ● Simple but effective method to stationarize a time series.
                ● Take difference of consecutive terms in a series.
                ● Known as a Difference Series of Order 1.
                    
                
            Test for Stationarity of Data (No Trend):
                
                ADF (Augmented Dickey Fuller Test):
                    H0 : Data is stationary
                    H1 : Data is non-Stationary
                    
                    If p value < 0.05 (ie, α), reject null hypothesis.
                    
            ARIMA Models:
            
                ● Assumptions:
                    ● Only Stationary Series can be forecasted!!
                    ● If Stationarity condition is violated, the first step is to stationarize the series
            
                ● Auto Regression (AR):
                    AR: Current observation is regressed on past observations
                        Y(t) = β1Y(t-1) + β2Y(t-2) + β3Y(t-3) + ... +βpY(t-p) + ε(t)
                
                    ● When value of a time series depends on its value at the previous time point
                    ● Various economic and financial series are autoregressive
                        ̶ GDP of a country
                        ̶ Stock prices
                        ̶ Consumption expenditure
                        
                    ● How many lags to consider?
                        ● Based on statistical significance (PACF Plot).
                        ● Lags are the features here.
                        
                    ● AutoCorrelation:
                        ● Autocorrelation: Correlation with self
                        ● Autocorrelation of different orders gives inside information regarding time series
                        ● Autocorrelations decreasing as lag increases
                        ● Autocorrelations significant till high order
                        ● Significant autocorrelations imply observations of long past influences current observation
                        ● Determines order p of the series
                            –1 ≤ ACF ≤ 1
                            ACF(0) = 1
                        ● ACF makes sense only if the series is stationary
                        
                        Ex:
                            ● Correlation between Original series and Lag(1) series = ACF(1)
                            ● Correlation between Original series and Lag(2) series = ACF(2)
                            ..
                    
                    ● Partial Autocorrelation:
                        ● Partial autocorrelation adjusts for the intervening periods
                        ● PACF(1) = ACF(1)
                        ● PACF(2) is the correlation between Original and Lag(2) series AFTER the influence of Lag(1) series has been eliminated
                        ● PACF(3) is the correlation between Original and Lag(3) series AFTER the influence of Lag(1) and Lag(2) series has been eliminated
                        ..
                        –1 ≤ PACF ≤ 1
                        
                    ● Reading ACF and PACF:
                        ● ACF and PACF together to be considered for identification of order of Autoregression
                        ● Seasonal ACF show significant correlation at seasonal points
                        
                        ● ACF:

                            ● ACF: Exp Decay				PACF : Spike at Lag 1			P: 1	Q: 0
                            ● ACF: Sine/Exp Decay			PACF : Spike at Lag 1 and 2		P: 2	Q: 0
                            ● ACF: Spike at Lag 1			PACF : Exp Decay				P: 0	Q: 1
                            ● ACF: Spike at Lag 1 and 2	PACF : Sine/Exp Decay			P: 0	Q: 2
                            ● ACF: Exp Decay at Lag 1		PACF : Exp Decay at Lag 1		P: 1	Q: 1

                    
                ● Moving Average for Residuals (MA):
                    ● Current observation is regressed on past forecast errors.
                        Y(t) = ε(t) + α1ε(t-1) +α2ε(t-2) + ... + αqε(t-q) 
                            where, |α1| < 1
                        α1, α2, ..., αq: Moving average parameters
                    
                ● ARMA: 
                    ● When Current observation is a linear combination of past observations and past white (random) noises
                        Y(t) = β1Y(t-1) + β2Y(t-2) + β3Y(t-3) + ... +βpY(t-p) + ε(t)
                            + α1ε(t-1) +α2ε(t-2) + ... + αqε(t-q) 

                    ● If the original series is not stationary, differencing is necessary
                    ● Most often differencing of order 1 makes the series stationary
                    ● But higher order differencing may be needed
                    ● Order = d
			
        """
        pass

    def st_1_eda_steps():
        """
        Read Data:
            df1 = pd.read_csv('AirPassenger.csv', parse_dates = ['Year-Month'], index_col = 'Year-Month')
        
        Make your own date range:
            date = pd.date_range(start='1/1/1956', end='1/1/1996', freq='M')
            #or
            quarters= pd.date_range(start='9/30/1982', end='3/31/1992', freq='Q')
            #or
            date = pd.date_range(start='1/1/2009', end='11/26/2010', freq='D')
            df2['Time_Stamp'] = pd.DataFrame(date)
            #For Business Days:
            from pandas.tseries.offsets import BDay
            date = pd.date_range(start='05/01/2017', end='01/31/2019', freq=BDay())
        
        Plot TIme Series:
            df1.plot(figsize=(12,8),grid=True)
        
        Handling Missing Values:
            Check:
                df4.isnull().sum()
            Impute Rolling Mean:
                ## imputing using rolling mean
                daily = df4.fillna(df4.rolling(6,min_periods=1).mean())
                ## imputing using interpolation
                df4_imputed= df4.interpolate(method = 'linear')

        Modifying Time Series Range:
            #Change Monthly Series to quarterly
            df1_q = df1.resample('Q').sum()

        Plot sales across years (with Pivot Table):
            months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October','November', 'December']
            yearly_sales_across_years = pd.pivot_table(df1, values = 'Pax', columns = df1.index.year,index = df1.index.month_name())
            yearly_sales_across_years = yearly_sales_across_years.reindex(index = months)
            yearly_sales_across_years.plot()
            plt.grid()
            plt.legend(loc='best');
        Monthly Plot for CO2 in ppm (with Pivot Table):
            monthly_co2ppm_across_years = pd.pivot_table(df, values = 'CO2 ppm', columns = df.index.year, index = df.index.month_name())
            monthly_co2ppm_across_years
            monthly_co2ppm_across_years.plot()
            plt.grid()
            plt.legend(loc='best');

        Boxplot for Variation:
            sns.boxplot(x=df.index.month,y=df['CO2 ppm'])
            plt.grid();

        Lag Plots:
            df['Month']=pd.date_range(start="01/01/1981",end="31/12/1993",freq='M')
            df=df.drop('Year',axis=1)
            df=df.set_index('Month')
            pd.plotting.lag_plot(df, lag=1)
            pd.plotting.lag_plot(df, lag=2)
            pd.plotting.lag_plot(df, lag=3)

        Decomposition Series:
            Additive:
                from statsmodels.tsa.seasonal import seasonal_decompose, STL
                decomposition = seasonal_decompose(df1,model='additive')
                decomposition.plot();
            Multiplicative:
                from statsmodels.tsa.seasonal import seasonal_decompose, STL
                decomposition = seasonal_decompose(df1,model='multiplicative')
                decomposition.plot();
            Decompose by losses:
                decomposition = STL(df1).fit()
                decomposition.plot();
            Decompose by Logloss:
                    decomposition = STL(np.log10(df1)).fit()
                    decomposition.plot();

        Plot Moving Average (rolling mean and std deviation):

            rolmean = df_final.rolling(window=15).mean()
            rolstd = df_final.rolling(window=15).std()
            orig = plt.plot(df_final, color='blue',label='Original')
            mean = plt.plot(rolmean, color='red', label='Rolling Mean')
            std = plt.plot(rolstd, color='black', label = 'Rolling Std')
            plt.legend(loc='best')
            plt.title('Rolling Mean & Standard Deviation')
            plt.show()
            
            For 5:
                plt.figure(figsize=(12,8))
                plt.plot(df5, label='closing price')
                plt.plot(df5.rolling(5).mean(), label='Moving Average')
                plt.legend(loc='best')
                plt.show()
            For 30:
                plt.figure(figsize=(12,8))
                plt.plot(df5, label='closing price')
                plt.plot(df5.rolling(30).mean(), label='Moving Average')
                plt.legend(loc='best')
                plt.show()
            

        Stationarity Test:
            # ADF Test..
            from statsmodels.tsa.stattools import adfuller
            observations= df.values
            test_result = adfuller(observations)
            test_result

        Plot ACF:
            from statsmodels.graphics.tsaplots import  plot_acf,plot_pacf
            plot_acf(df,lags=30);
        
        Plot PACF:
            from statsmodels.graphics.tsaplots import  plot_acf,plot_pacf
            plot_pacf(df,lags=30);

        Train-Test-Split of Time Series:
            train_end=datetime(1971,9,30)
            test_end=datetime(1972,9,30)
            train = df[:train_end] 
            test = df[train_end + timedelta(days=1):test_end]
            display(train)
            display(test)

        Plot Differencing:
            1st Order:
                df_1=df.diff(periods=1).dropna()
                df_1.plot(title='1st oder differencing')
                #Check Stationarity Again:
                observations= df_1.values
                test_result = adfuller(observations)
                test_result
            Seasonal Differencing:
                df2_12=df2.diff(periods=12).dropna()
                df2_12.plot()
                #Check Stationarity Again:
                observations= df2_12.values
                test_result = adfuller(observations)
                test_result

        """
        pass

    def st_2_model_build_steps():
        """
        Exponential Smoothing Models:

            SES:
                from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
                model_SES = SimpleExpSmoothing(train,initialization_method='estimated')
                model_SES_fit1 = model_SES.fit(optimized=True)
                SES_predict1 = model_SES_fit1.forecast(steps=len(test))
                plt.plot(train, label='Train')
                plt.plot(test, label='Test')
                plt.plot(SES_predict1,label='forecast')
                plt.legend(loc='best')
                plt.grid()

            DES:
                from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
                model_DES = Holt(train,exponential=True, initialization_method='estimated')
                model_DES_fit1 = model_DES.fit(optimized=True)
                model_DES_fit1.summary()
                DES_predict1 = model_DES_fit1.forecast(steps=len(test))
                #Plot:
                plt.plot(train, label='Train')
                plt.plot(test, label='Test')
                plt.plot(DES_predict1, label='DES forecats')
                plt.legend(loc='best')
                plt.grid()

            TES:
                from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

                #Additive Model:
                model_TES_add = ExponentialSmoothing(train,trend='additive',seasonal='additive',initialization_method='estimated')
                model_TES_add = model_TES_add.fit(optimized=True)
                model_TES_add.summary()
                TES_add_predict =  model_TES_add.forecast(len(test))
                plt.plot(train, label='Train')
                plt.plot(test, label='Test')
                plt.plot(TES_add_predict, label='forecast')
                plt.legend(loc='best')
                plt.grid()
                #Metrics:
                mean_squared_error(test.values,TES_add_predict.values,squared=False)
                def MAPE(y_true, y_pred):
                    return np.mean((np.abs(y_true-y_pred))/(y_true))*100
                MAPE(test['Sales'],TES_add_predict)

                #Multiplicative Model:
                model_TES_mul = ExponentialSmoothing(train,trend='multiplicative',seasonal='multiplicative',initialization_method='estimated')
                model_TES_mul = model_TES_mul.fit(optimized=True)
                model_TES_mul.summary()
                TES_mul_predict =  model_TES_mul.forecast(len(test))
                plt.plot(train, label='Train')
                plt.plot(test, label='Test')
                plt.plot(TES_mul_predict, label='TES forecast')
                plt.legend(loc='best')
                plt.grid()
                #Metrics:
                mean_squared_error(test.values,TES_mul_predict.values,squared=False)
                def MAPE(y_true, y_pred):
                    return np.mean((np.abs(y_true-y_pred))/(y_true))*100
                MAPE(test['Pax'],TES_mul_predict)

        ARIMA Models:

            AR Process with Simulated Data:
                AR(1):
                    from statsmodels.tsa.arima_process import ArmaProcess
                    ar = np.array([1,-0.33])
                    ma = np.array([1])
                    object1 = ArmaProcess(ar, ma)
                    simulated_data_1 = object1.generate_sample(nsample=150)
                    df1=pd.Series(simulated_data_1)
                    #Plot
                    df1.plot()
                    plot_acf(df1)
                    plot_pacf(df1)
                AR(2):
                    from statsmodels.tsa.arima_process import ArmaProcess
                    ar = np.array([1,-0.33,-0.5])
                    ma = np.array([1])
                    object1 = ArmaProcess(ar, ma)
                    simulated_data_1 = object1.generate_sample(nsample=150)
                    df2=pd.Series(simulated_data_1)
                    #Plot
                    df2.plot()
                    plot_acf(df2)
                    plot_pacf(df2)
                MA(2):
                    from statsmodels.tsa.arima_process import ArmaProcess
                    ar = np.array([1])
                    ma = np.array([1,0.7,0.2])
                    object1 = ArmaProcess(ar, ma)
                    simulated_data_1 = object1.generate_sample(nsample=600)
                    df2=pd.Series(simulated_data_1)
                    #Plot
                    df2.plot()
                    plot_acf(df2)
                    plot_pacf(df2)

            ARMA Model:

                #Split:
                train_end=datetime(2002,12,31)
                test_end=datetime(2004,12,31)
                train = df[:train_end] 
                test = df[train_end + timedelta(days=1):test_end]
                model=ARIMA(endog=train,order=(2,0,2))
                model_fit=model.fit()
                print(model_fit.summary())
                #Prediction
                pred_start=test.index[0]
                pred_end=test.index[-1]
                predictions=model_fit.predict(start=pred_start, end=pred_end)
                predictions1=model_fit.forecast(12)
                #Plot Forecast:
                plt.plot(train,label='Training Data')
                plt.plot(test,label='Test Data')
                plt.plot(test.index,predictions,label='Predicted Data - ARMA(1,0)')
                plt.legend(loc='best')
                plt.grid();
                #Find Residuals (usually with stocks):
                residuals = test.growth - predictions
                plt.plot(residuals)
                plt.show()
                #Metrics:
                from sklearn.metrics import mean_squared_error
                mean_squared_error(test.values,predictions.values,squared=False)
                qqplot(residuals,line="s");

            ARIMA Model:

                model=ARIMA(endog=train,order=(1,0,0))
                model_fit=model.fit()
                print(model_fit.summary())
                #Predict:
                pred_start=test.index[0]
                pred_end=test.index[-1]
                pred_end
                forecast=model_fit.forecast(10)
                predictions=model_fit.predict(start=pred_start, end=pred_end)
                plt.plot(train,label='Training Data')
                plt.plot(test,label='Test Data')
                plt.plot(test.index,predictions,label='Predicted Data - ARMA(1,0)')
                plt.legend(loc='best')
                plt.grid();
                #Find Residuals (usually with stocks):
                residuals = test.Close - predictions
                plt.plot(residuals)
                plt.show()#Metrics:
                from sklearn.metrics import mean_squared_error
                mean_squared_error(test.values,predictions.values,squared=False)
                qqplot(residuals,line="s");

            ARIMA Model - Selecting Order with Lowest AIC:

                #TTSplit:
                train_end=datetime(1978,12,1)
                test_end=datetime(1980,12,1)
                train = df[:train_end] 
                test = df[train_end + timedelta(days=1):test_end]
                #Select AIC:
                import itertools
                p = q = range(0, 4)
                d= range(1,2)
                pdq = list(itertools.product(p, d, q))
                print('parameter combinations for the Model')
                for i in range(1,len(pdq)):
                    print('Model: {}'.format(pdq[i]))
                dfObj1 = pd.DataFrame(columns=['param', 'AIC'])
                dfObj1
                for param in pdq:
                    try:
                        mod = ARIMA(train, order=param)
                        results_Arima = mod.fit()
                        print('ARIMA{} - AIC:{}'.format(param, results_Arima.aic))
                        dfObj1 = dfObj1.append({'param':param, 'AIC': results_Arima.aic}, ignore_index=True)
                    except:
                        continue
                dfObj1.sort_values(by=['AIC'])
                #Build Model:
                model = ARIMA(train, order=(2,1,1))
                results_Arima = model.fit()
                print(results_Arima.summary())
                #Predict (only the differences):
                pred_start=test.index[0]
                pred_end=test.index[-1]
                ARIMA_predictions=results_Arima.predict(start=pred_start, end=pred_end)
                ARIMA_predictions
                ARIMA_pred=ARIMA_predictions.cumsum()
                ARIMA_pred
                #Forecast (Final value with differences):
                predict_fc = ARIMA_pred.copy()
                columns = train.columns
                for col in columns:
                        predict_fc[str(col)+'_forecast'] = train[col].iloc[-1] + predict_fc[str(col)]
                #Plot Forecast:
                plt.plot(train,label='Training Data')
                plt.plot(test,label='Test Data')
                plt.plot(test.index,predict_fc['CO2 ppm_forecast'],label='Predicted Data - ARIMA')
                plt.legend(loc='best')
                plt.grid();
                residuals = test['CO2 ppm'] - predict_fc['CO2 ppm_forecast']
                qqplot(residuals,line="s");
                #Metrics:
                from sklearn.metrics import  mean_squared_error
                rmse = mean_squared_error(test['CO2 ppm'],predict_fc['CO2 ppm_forecast'], squared=False)
                print(rmse)
                def MAPE(y_true, y_pred):
                    return np.mean((np.abs(y_true-y_pred))/(y_true))*100
                mape=MAPE(test['CO2 ppm'].values,predict_fc['CO2 ppm_forecast'].values)
                print(mape)

            SARIMA Model - Selecting Order with Lowest AIC:

                #Select AIC:
                import itertools
                p = q = range(0, 3)
                d= range(1,2)
                pdq = list(itertools.product(p, d, q))
                model_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
                print('Examples of parameter combinations for Model...')
                print('Model: {}{}'.format(pdq[1], model_pdq[1]))
                print('Model: {}{}'.format(pdq[1], model_pdq[2]))
                print('Model: {}{}'.format(pdq[2], model_pdq[3]))
                print('Model: {}{}'.format(pdq[2], model_pdq[4]))
                dfObj2 = pd.DataFrame(columns=['param','seasonal', 'AIC'])
                dfObj2
                import statsmodels.api as sm
                for param in pdq:
                    for param_seasonal in model_pdq:
                        mod = sm.tsa.statespace.SARIMAX(train,
                                                            order=param,
                                                            seasonal_order=param_seasonal,
                                                            enforce_stationarity=False,
                                                            enforce_invertibility=False)
                        results_SARIMA = mod.fit()
                        print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results_SARIMA.aic))
                        dfObj2 = dfObj2.append({'param':param,'seasonal':param_seasonal ,'AIC': results_SARIMA.aic}, ignore_index=True)
                dfObj2.sort_values(by=['AIC'])
                #Build Model:
                model = sm.tsa.statespace.SARIMAX(train,
                                order=(1,1,0),
                                seasonal_order=(1,1,2,12),
                                )
                model_Sarima = model.fit()
                print(model_Sarima.summary())
                #Predict (only the differences):
                SARIMA_predictions=model_Sarima.predict(start=pred_start, end=pred_end)
                plt.plot(train,label='Training Data')
                plt.plot(test,label='Test Data')
                plt.plot(test.index,SARIMA_predictions,label='Predicted Data - SARIMA')
                plt.legend(loc='best')
                plt.grid();
                #Forecast (Final value with differences):
                predict_fc = ARIMA_pred.copy()
                columns = train.columns
                for col in columns:
                        predict_fc[str(col)+'_forecast'] = train[col].iloc[-1] + predict_fc[str(col)]
                #Plot Forecast:
                plt.plot(train,label='Training Data')
                plt.plot(test,label='Test Data')
                plt.plot(test.index,predict_fc['CO2 ppm_forecast'],label='Predicted Data - ARIMA')
                plt.legend(loc='best')
                plt.grid();
                residuals = test['CO2 ppm'] - predict_fc['CO2 ppm_forecast']
                qqplot(residuals,line="s");
                #Metrics:
                from sklearn.metrics import  mean_squared_error
                rmse = mean_squared_error(test['CO2 ppm'],SARIMA_predictions, squared=False)
                print(rmse)
                def MAPE(y_true, y_pred):
                    return np.mean((np.abs(y_true-y_pred))/(y_true))*100
                mape = MAPE(test['CO2 ppm'],SARIMA_predictions)
                print(mape)
                model_Sarima.plot_diagnostics(figsize=(16, 8))
                plt.show()

                #Build Model on Full Data:
                model = sm.tsa.statespace.SARIMAX(df,
                                order=(1,1,0),
                                seasonal_order=(1,1,2,12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                model_Sarima = model.fit()
                print(model_Sarima.summary())
                #Forecast (Final value with differences):
                pred95 = model_Sarima.get_forecast(steps=24)
                pred95 = pred95.conf_int()
                pred95
                #Plot Forecast:
                axis = df.plot(label='Observed', figsize=(15, 8))
                forecast.plot(ax=axis, label='Forecast', alpha=0.7)
                axis.fill_between(forecast.index, pred95['lower CO2 ppm'], pred95['upper CO2 ppm'], color='k', alpha=.15)
                axis.set_xlabel('Year-Months')
                axis.set_ylabel('CO2 ppm')
                plt.legend(loc='best')
                plt.show()

            VAR Model (Too Complex, not asked..)
        """
        pass

