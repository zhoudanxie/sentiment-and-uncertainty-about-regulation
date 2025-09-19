clear
graph drop _all
cap drop all

*Basic data set-up
use data/processed_data/regindex_area,clear

***Specify index of interest
gen index=$index

*Michigan consumer sentiment data - some robustness test data
merge 1:1 year month using data/processed_data/consumer_sentiment_data,keep(1 3) nogen
ren Month mich

***Merge in Stock data
merge m:1 year month using data/processed_data/sp_500_data, keep(1 3) nogen

***Merge in Macro data 
merge m:1 year month using data/processed_data/macro_data, keep(1 3) nogen
replace vix=. if year<1990

***Merge in BBD EPU 
merge 1:1 year month using data/processed_data/epu,keep(1 3) nogen

***Merge in Shapiro sentiment
merge 1:1 year month using data/processed_data/newssent,keep(1 3) nogen

***Transform data
gen lgdp=log(gdp)

gen ym=year+(month-1)/12
replace quarter=floor((month+2)/3) if quarter==.
gen yq=year+(quarter-1)/4
gen time=ym*12
tsset time
gen lemp=log(employment)
gen lip=log(indus)
gen lsp=log(sp)
gen lvix=log(vix)

ren sp sp
ren fedfundsrate ffr

*For monthly, output is industrial production
replace lgdp=lip

* Choose impulse response horizon
local hmax = 12

* Define a shock
sum index
global ratio=-r(sd)

* Select lags
varsoc index lsp ffr lemp lgdp, maxlag(10)

/* Generate LHS variables for the LPs */

* levels
forvalues h = 0/`hmax' {
	gen lgdp_`h' = f`h'.lgdp 
}

forvalues h = 0/`hmax' {
	gen lemp_`h' = f`h'.lemp 
}

/* Run the LPs */
* Levels - lgdp
eststo clear
cap drop b u90 d90 u95 d95 Years Zero
gen Years = _n-1 if _n<=`hmax'+1
gen Zero =  0    if _n<=`hmax'+1
gen b=0
gen u90=0
gen d90=0
gen u95=0
gen d95=0
forv h = 0/`hmax' {
	* levels
	 reg lgdp_`h' l(0/3).index l(0/3).lsp l(0/3).ffr l(0/3).lemp l(0/3).lgdp , vce(robust)
replace b = _b[index]*$ratio*100                    if _n == `h'+1
replace u90 = (_b[index] + 1.645* _se[index])*$ratio*100  if _n == `h'+1
replace d90 = (_b[index] - 1.645* _se[index])*$ratio*100  if _n == `h'+1
replace u95 = (_b[index] + 1.96* _se[index])*$ratio*100  if _n == `h'+1
replace d95 = (_b[index] - 1.96* _se[index])*$ratio*100  if _n == `h'+1
eststo
}

preserve
keep Years b u90 d90 u95 d95
keep if Years!=.
save "output/${index}_lgdp.dta",replace
restore

* Levels - lemp
eststo clear
cap drop b u90 d90 u95 d95 Years Zero
gen Years = _n-1 if _n<=`hmax'+1
gen Zero =  0    if _n<=`hmax'+1
gen b=0
gen u90=0
gen d90=0
gen u95=0
gen d95=0
forv h = 0/`hmax' {
	* levels
	 reg lemp_`h' l(0/3).index l(0/3).lsp l(0/3).ffr l(0/3).lemp l(0/3).lgdp, vce(robust)
replace b = _b[index]*$ratio*100                    if _n == `h'+1
replace u90 = (_b[index] + 1.645* _se[index])*$ratio*100  if _n == `h'+1
replace d90 = (_b[index] - 1.645* _se[index])*$ratio*100  if _n == `h'+1
replace u95 = (_b[index] + 1.96* _se[index])*$ratio*100  if _n == `h'+1
replace d95 = (_b[index] - 1.96* _se[index])*$ratio*100  if _n == `h'+1
eststo
}

preserve
keep Years b u90 d90 u95 d95
keep if Years!=.
save "output/${index}_lemp.dta",replace
restore
 
/* THE END */
