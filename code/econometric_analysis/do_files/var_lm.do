clear all
set more off

************************************
************************************
global lags=3
global lags2=2*$lags
global lags3=4*$lags
global jump=6
global plot=60/$months
************************************
************************************

*Basic data set-up
use data/processed_data/data_for_analysis/regindex,clear

*Michigan consumer sentiment data - some robustness test data
merge 1:1 year month using data/processed_data/data_for_analysis/consumer_sentiment_data,keep(1 3) nogen
ren Month mich

***Merge in NIPA data
merge m:1 year quarter using data/processed_data/data_for_analysis/nipa,keep(1 3) nogen

***Merge in Stock data
merge m:1 year month using data/processed_data/data_for_analysis/sp_500_data, keep(1 3) nogen

***Merge in Macro data 
merge m:1 year month using data/processed_data/data_for_analysis/macro_data, keep(1 3) nogen
replace vix=. if year<1990

***Merge in BBD EPU 
merge 1:1 year month using data/processed_data/data_for_analysis/epu,keep(1 3) nogen

***Merge in Shapiro sentiment
merge 1:1 year month using data/processed_data/data_for_analysis/newssent,keep(1 3) nogen

***Transform data
gen lgdp=log(gdp)

gen ym=year+(month-1)/12
replace quarter=floor((month+2)/3) if quarter==.
gen yq=year+(quarter-1)/4
gen time=ym*12
tsset time
gen lemp=log(employment)
gen lip=log(indus)

ren sp sp
gen lsp=log(sp)
gen lgross=log(gross)
gen lvix=log(vix)

ren fedfundsrate ffr

***Label variables
lab var sp "Closing value of S&P500 index"
lab var gdp "Real Gross Domestic Product (GDP)"
lab var vix "VIX index of implied equity volatility on the S&P500"
lab var ym "Year in fraction (i.e. 2010.5 is July 2010)"
lab var year "Year"
lab var month "Calendar Month"
lab var time "Indicator for Stata to tsset, definied as ym*12"
lab var employment "Aggregate non-farm employment"
lab var lemp "Log employment"
lab var lgdp "Log real gdp"
lab var lip "Log industrial production"
lab var lsp "Log stock index"
lab var lgross "Log gross investment"

*For Quarterly Collapse the Data
if $months==3 {
	// Use quarterly estimates
	replace rpu=rpu_qr
	replace lm=lm_qr
	replace gi=gi_qr
	replace lsd=lsd_qr

	// Collapse to quarterly means
	collapse rri rpu gi lsd lm epu newssent lsp ffr lemp lgdp lgross lip vix lvix year mich quarter,by(yq)
	gen time=4*yq
	tsset time
	gen ym=yq 
}
gen linvest=lgross

*For monthly, output is industrial production
if $months==1 {
	*Note to save changing variable names too much call industrial production GDP for monthly runs on IP
	replace lgdp=lip
}

tsset time
gen dlgdp=lgdp-l.lgdp
forval i=1(1)3 {
gen l`i'=l`i'.dlgdp
gen f`i'=f`i'.dlgdp
}

* Define a shock
su lm if year==2005|year==2006
global lm_low=r(mean)
su lm if year==2011
global lm_high=r(mean)

sum lm
global lm_sd=r(sd)

* Recessions
gen recession=0
replace recession=1 if (yq>=1945)&(yq<=1945.76)
replace recession=1 if (yq>=1948.74)&(yq<=1949.76)
replace recession=1 if (yq>=1953.24)&(yq<=1954.26)
replace recession=1 if (yq>=1957.49)&(yq<=1958.26)
replace recession=1 if (yq>=1960.24)&(yq<=1961)
replace recession=1 if (yq>=1969.74)&(yq<=1970.76)
replace recession=1 if (yq>=1973.74)&(yq<=1975)
replace recession=1 if (yq>=1980)&(yq<=1980.51)
replace recession=1 if (yq>=1981.49)&(yq<=1982.76)
replace recession=1 if (yq>=1990.49)&(yq<=1991)
replace recession=1 if (yq>=2001)&(yq<=2001.76)
replace recession=1 if (yq>=2008)&(yq<=2009.26)

*For magnitudes note investment falls in post-WWII recessions 
reg lgross recession yq
reg lemp recession yq
sa output/var,replace

*Baseline regressions
u output/var,replace
irf set output/var_results,replace
gen ym2=ym^2

qui var lm lsp ffr lemp lgdp,lags(1(1)$lags)
irf create baseline,step($plot) replace set(output/var_results)

*Robustness checks
if $months==3 {
//quarterly VAR with investment
cap qui var lm lsp ffr lgross lgdp,lags(1(1)$lags)
cap irf create quarterly,step($plot) replace set(output/var_results)
}

//add time trend
cap qui var lm lsp ffr lemp lgdp ym,lags(1(1)$lags)
cap irf create timetrend,step($plot) replace set(output/var_results)

//bivariate VAR
qui var lm lgdp,lags(1(1)$lags)	
irf create bi_output,step($plot) replace set(output/var_results)

//bivariate VAR with reverse ordering
qui var lgdp lm,lags(1(1)$lags)
irf create rbi_output,step($plot) replace set(output/var_results)

//bivariate VAR
qui var lm lemp,lags(1(1)$lags)	
irf create bi_emp,step($plot) replace set(output/var_results)

//bivariate VAR with reverse ordering
qui var lemp lm,lags(1(1)$lags)
irf create rbi_emp,step($plot) replace set(output/var_results)

//drop log S&P500
qui var lm ffr lemp lgdp,lags(1(1)$lags)
irf create nsp,step($plot) replace set(output/var_results)

//reverse ordering
qui var lgdp lemp ffr lsp lm,lags(1(1)$lags)
irf create reverse,step($plot) replace set(output/var_results)

//add VIX
qui var lm vix lsp ffr lemp lgdp,lags(1(1)$lags)
irf create vix,step($plot) replace set(output/var_results)

//add log(VIX)
qui var lm lvix lsp ffr lemp lgdp,lags(1(1)$lags)
irf create lvix,step($plot) replace set(output/var_results)

//six lags
qui var lm lsp ffr lemp lgdp,lags(1(1)$lags2)
irf create lags6,step($plot) replace set(output/var_results)

//12 lags
qui var lm lsp ffr lemp lgdp,lags(1(1)$lags3)
irf create lags12,step($plot) replace set(output/var_results)

//add Michigan index and order second
qui var lm mich lsp ffr lemp lgdp,lags(1(1)$lags)
irf create mich_second,step($plot) replace set(output/var_results)

//add Michigan index and order first
qui var mich lm lsp ffr lemp lgdp,lags(1(1)$lags)
irf create mich_first,step($plot) replace set(output/var_results)

//add BBD EPU and order second
qui var lm epu lsp ffr lemp lgdp,lags(1(1)$lags)
irf create epu_second,step($plot) replace set(output/var_results)

//add BBD EPU and order first
qui var epu lm lsp ffr lemp lgdp,lags(1(1)$lags)
irf create epu_first,step($plot) replace set(output/var_results)

//add Shapiro news sentiment and order second
qui var lm newssent lsp ffr lemp lgdp,lags(1(1)$lags)
irf create newssent_second,step($plot) replace set(output/var_results)

//add Shapiro news sentiment and order first
qui var newssent lm lsp ffr lemp lgdp,lags(1(1)$lags)
irf create newssent_first,step($plot) replace set(output/var_results)

//drop covid period
qui var lm lsp ffr lemp lgdp if year<2020,lags(1(1)$lags)
irf create no_covid,step($plot) replace set(output/var_results)

//before great recession
qui var lm lsp ffr lemp lgdp if year<2007,lags(1(1)$lags)
irf create before_great,step($plot) replace set(output/var_results)

//IRFs over a longer time horizon
global plot2=$plot*10
qui var lm lsp ffr lemp lgdp,lags(1(1)$lags)
irf create baseline_long,step($plot2) replace set(output/var_results)

// baseline starting 1995
qui var lm lsp ffr lemp lgdp if year>1994,lags(1(1)$lags)
irf create baseline1995,step($plot) replace set(output/var_results)


***CLEAN IRF OUTPUT
u output/var_results.irf,replace
ren step _step
drop cirf coirf sirf  fevd sf* mse* dm cdm stddm
drop stdi* stdc* stdf* stds*

gen ir=impulse+response
drop impulse response
ren _step step

qui reshape wide oirf irf stdoirf,i(step irfname) j(ir) string


***THREE STEPS OF NORMALIZATION TO GET MAGNITUDES
****Normalizing the oirfs into the same units as irf
egen lmratio=max((step==0)*oirflmlm),by(irfname)

replace oirflmlgdp=oirflmlgdp/lmratio
replace oirflmlemp=oirflmlemp/lmratio
cap replace oirflmlgross=oirflmlgross/lmratio
replace oirflmlvix=oirflmlvix/lmratio
replace oirflmvix=oirflmvix/lmratio

replace stdoirflmlgdp=stdoirflmlgdp/lmratio
replace stdoirflmlemp=stdoirflmlemp/lmratio
cap replace stdoirflmlgross=stdoirflmlgross/lmratio
replace stdoirflmlvix=stdoirflmlvix/lmratio
replace stdoirflmvix=stdoirflmvix/lmratio

****Normalizing the oirfs into meaningful units
global ratio=-$lm_sd
global se=1.645
global se95=1.96
foreach var in lgdp lemp lgross lvix {
	cap replace oirflm`var'=oirflm`var'*$ratio
	cap replace stdoirflm`var'=stdoirflm`var'*$ratio
	****Normalizing the impact into % unit
	cap replace oirflm`var'=oirflm`var'*100
	cap replace stdoirflm`var'=stdoirflm`var'*100
	***Generate standard error bands
	cap gen l`var'=oirflm`var'-stdoirflm`var'*$se
	cap gen h`var'=oirflm`var'+stdoirflm`var'*$se
	cap gen l`var'95=oirflm`var'-stdoirflm`var'*$se95
	cap gen h`var'95=oirflm`var'+stdoirflm`var'*$se95	
}

foreach var in vix {
	cap replace oirflm`var'=oirflm`var'*$ratio
	cap replace stdoirflm`var'=stdoirflm`var'*$ratio
	***Generate standard error bands
	cap gen l`var'=oirflm`var'-stdoirflm`var'*$se
	cap gen h`var'=oirflm`var'+stdoirflm`var'*$se
	cap gen l`var'95=oirflm`var'-stdoirflm`var'*$se95
	cap gen h`var'95=oirflm`var'+stdoirflm`var'*$se95	
}

lab var step "year"

****Export output
if $months==1{
save output/var_lm,replace
}
if $months==3 {
save output/var_lm_quarterly,replace
}
