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
        pass

    def st_2_model_build_steps():
        pass

